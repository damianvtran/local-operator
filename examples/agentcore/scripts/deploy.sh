#!/usr/bin/env bash
# Deploy Local Operator to Amazon Bedrock AgentCore Runtime.
#
#   scripts/deploy.sh --profile <aws-profile> [--region <region>] [--stack-name lop-agentcore]
#                     [--model-id <bedrock model or inference profile id>]
#                     [--lop-version <pypi version>] [--approvals auto|refuse|tool,list]
#
# Phases (each idempotent, so a re-run resumes where a failure stopped):
#   1. <stack>-build    ECR repository, build-context bucket, arm64 CodeBuild project
#   2. image            zip app/, upload, CodeBuild docker build+push (skipped when
#                       the content-addressed tag already exists in ECR)
#   3. <stack>-runtime  execution role + AWS::BedrockAgentCore::Runtime
#
# --profile is REQUIRED on purpose: this script creates billable resources and an
# IAM role, so it never falls back to whatever credentials happen to be ambient.
# Nothing here assumes an account, region or organisation.
set -euo pipefail

usage() {
  sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
}

PROFILE=""
REGION=""
STACK="lop-agentcore"
# A broadly available Claude inference profile. Check which ids your account and
# region offer with: aws bedrock list-inference-profiles --profile <p> --region <r>
MODEL_ID="global.anthropic.claude-sonnet-5-5"
LOP_VERSION="0.92.1"
APPROVALS="auto"
TOOLS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --region) REGION="${2:-}"; shift 2 ;;
    --stack-name) STACK="${2:-}"; shift 2 ;;
    --model-id) MODEL_ID="${2:-}"; shift 2 ;;
    --lop-version) LOP_VERSION="${2:-}"; shift 2 ;;
    --approvals) APPROVALS="${2:-}"; shift 2 ;;
    --tools) TOOLS="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$PROFILE" ]]; then
  echo "error: --profile is required (the AWS CLI profile to deploy with); refusing to use ambient credentials." >&2
  exit 2
fi
if [[ -z "$REGION" ]]; then
  REGION="$(aws configure get region --profile "$PROFILE" || true)"
fi
if [[ -z "$REGION" ]]; then
  echo "error: --region is required (profile '$PROFILE' has no default region)." >&2
  exit 2
fi
if [[ ! "$STACK" =~ ^[a-z][a-z0-9-]{1,40}$ ]]; then
  echo "error: --stack-name must be lowercase letters, digits and hyphens (2-41 chars)." >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AWS=(aws --profile "$PROFILE" --region "$REGION")
BUILD_STACK="${STACK}-build"
RUNTIME_STACK="${STACK}-runtime"
# AgentCore runtime names allow letters, digits and underscores only.
RUNTIME_NAME="$(echo "$STACK" | tr '-' '_')"
# The foundation model behind a system inference profile is its id without the
# geo prefix (us., eu., apac., global., ...); a bare model id maps to itself.
FOUNDATION_MODEL_ID="$(echo "$MODEL_ID" | sed -E 's/^(us|us-gov|eu|apac|ap|ca|jp|au|global)\.//')"

stack_output() { # stack key
  "${AWS[@]}" cloudformation describe-stacks --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}

echo "==> Checking credentials for profile '$PROFILE' in $REGION"
"${AWS[@]}" sts get-caller-identity --query Arn --output text >/dev/null

echo "==> [1/3] Deploying build stack $BUILD_STACK"
"${AWS[@]}" cloudformation deploy \
  --stack-name "$BUILD_STACK" \
  --template-file "$HERE/infra/build.yaml" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides "ProjectName=$STACK" \
  --no-fail-on-empty-changeset

REPO_URI="$(stack_output "$BUILD_STACK" RepositoryUri)"
REPO_ARN="$(stack_output "$BUILD_STACK" RepositoryArn)"
BUCKET="$(stack_output "$BUILD_STACK" SourceBucketName)"
PROJECT="$(stack_output "$BUILD_STACK" BuildProjectName)"

# Content-addressed tag: same app/ + same lop version => same tag => no rebuild.
# ECR tags are immutable, so a tag always names exactly one build.
APP_HASH="$(cd "$HERE/app" && find . -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort \
  | xargs cat | { cat; printf '%s' "$LOP_VERSION"; } | shasum -a 256 | cut -c1-12)"
IMAGE_TAG="lop-${LOP_VERSION}-${APP_HASH}"
IMAGE_URI="${REPO_URI}:${IMAGE_TAG}"

echo "==> [2/3] Image $IMAGE_TAG"
if "${AWS[@]}" ecr describe-images --repository-name "${REPO_URI#*/}" \
     --image-ids "imageTag=$IMAGE_TAG" >/dev/null 2>&1; then
  echo "    already in ECR; skipping build"
else
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  # Python's zipfile is used instead of `zip`, which is not installed everywhere.
  python3 - "$HERE/app" "$WORK/context.zip" <<'PY'
import pathlib, sys, zipfile
root, out = pathlib.Path(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            zf.write(path, path.relative_to(root).as_posix())
PY
  KEY="contexts/${IMAGE_TAG}.zip"
  "${AWS[@]}" s3 cp --only-show-errors "$WORK/context.zip" "s3://$BUCKET/$KEY"
  BUILD_ID="$("${AWS[@]}" codebuild start-build --project-name "$PROJECT" \
    --source-location-override "$BUCKET/$KEY" \
    --environment-variables-override \
      "name=IMAGE_TAG,value=$IMAGE_TAG,type=PLAINTEXT" \
      "name=LOP_VERSION,value=$LOP_VERSION,type=PLAINTEXT" \
    --query build.id --output text)"
  echo "    CodeBuild $BUILD_ID"
  # Poll rather than block on a waiter: CodeBuild has none. Measured builds of
  # this image took about one minute each (n=3, 56-60 s, arm64 SMALL).
  while :; do
    STATUS="$("${AWS[@]}" codebuild batch-get-builds --ids "$BUILD_ID" \
      --query 'builds[0].buildStatus' --output text)"
    [[ "$STATUS" != "IN_PROGRESS" ]] && break
    sleep 15
  done
  if [[ "$STATUS" != "SUCCEEDED" ]]; then
    echo "error: image build $BUILD_ID finished $STATUS. Logs:" >&2
    "${AWS[@]}" codebuild batch-get-builds --ids "$BUILD_ID" \
      --query 'builds[0].logs.deepLink' --output text >&2
    exit 1
  fi
  echo "    pushed $IMAGE_URI"
fi

echo "==> [3/3] Deploying runtime stack $RUNTIME_STACK (model $MODEL_ID)"
"${AWS[@]}" cloudformation deploy \
  --stack-name "$RUNTIME_STACK" \
  --template-file "$HERE/infra/runtime.yaml" \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    "RuntimeName=$RUNTIME_NAME" \
    "ImageUri=$IMAGE_URI" \
    "RepositoryArn=$REPO_ARN" \
    "ModelId=$MODEL_ID" \
    "FoundationModelId=$FOUNDATION_MODEL_ID" \
    "Approvals=$APPROVALS" \
    "Tools=$TOOLS" \
  --no-fail-on-empty-changeset

RUNTIME_ARN="$(stack_output "$RUNTIME_STACK" AgentRuntimeArn)"
echo
echo "Deployed. Runtime ARN: $RUNTIME_ARN"
echo "Try it:"
echo "  python3 scripts/invoke.py --profile $PROFILE --region $REGION --stack-name $STACK \"What can you do?\""
