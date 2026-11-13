#!/usr/bin/env bash
# Delete everything deploy.sh created: the runtime stack first (it references the
# image), then the build stack (ECR repository with its images, build bucket,
# CodeBuild project).
#
#   scripts/teardown.sh --profile <aws-profile> [--region <region>] [--stack-name lop-agentcore]
#
# --profile is required for the same reason as in deploy.sh: deleting resources
# with whatever credentials happen to be ambient is how the wrong account loses
# a stack.
set -euo pipefail

PROFILE=""
REGION=""
STACK="lop-agentcore"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --region) REGION="${2:-}"; shift 2 ;;
    --stack-name) STACK="${2:-}"; shift 2 ;;
    -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$PROFILE" ]]; then
  echo "error: --profile is required; refusing to use ambient credentials." >&2
  exit 2
fi
[[ -z "$REGION" ]] && REGION="$(aws configure get region --profile "$PROFILE" || true)"
if [[ -z "$REGION" ]]; then
  echo "error: --region is required (profile '$PROFILE' has no default region)." >&2
  exit 2
fi
AWS=(aws --profile "$PROFILE" --region "$REGION")

for suffix in runtime build; do
  name="${STACK}-${suffix}"
  if "${AWS[@]}" cloudformation describe-stacks --stack-name "$name" >/dev/null 2>&1; then
    if [[ "$suffix" == build ]]; then
      # CloudFormation cannot delete a non-empty bucket; build contexts are
      # disposable, so empty it first.
      bucket="$("${AWS[@]}" cloudformation describe-stacks --stack-name "$name" \
        --query "Stacks[0].Outputs[?OutputKey=='SourceBucketName'].OutputValue" --output text)"
      if [[ -n "$bucket" && "$bucket" != "None" ]]; then
        "${AWS[@]}" s3 rm --only-show-errors --recursive "s3://$bucket"
      fi
    fi
    echo "==> Deleting $name"
    "${AWS[@]}" cloudformation delete-stack --stack-name "$name"
    "${AWS[@]}" cloudformation wait stack-delete-complete --stack-name "$name"
  else
    echo "==> $name not found; skipping"
  fi
done
echo "Teardown complete."
