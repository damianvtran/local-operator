#!/usr/bin/env bash
# Delete everything deploy.sh created: the runtime stack first (it references the
# image), then the build stack (ECR repository with its images, build bucket,
# CodeBuild project), then the two log groups the services create outside
# CloudFormation (the runtime's and the CodeBuild project's).
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
# A flag given without its value must fail loudly, not as a silent `shift 2`
# under `set -e`.
need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "error: $1 needs a value" >&2
    exit 2
  fi
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) need_value "$@"; PROFILE="$2"; shift 2 ;;
    --region) need_value "$@"; REGION="$2"; shift 2 ;;
    --stack-name) need_value "$@"; STACK="$2"; shift 2 ;;
    -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# Fail on credentials before anything else, as deploy.sh does: an expired SSO
# session or a wrong profile must not read as "nothing to delete".
echo "==> Checking credentials for profile '$PROFILE' in $REGION"
"${AWS[@]}" sts get-caller-identity --query Arn --output text >/dev/null

# describe_stack <name>: 0 = exists (prints nothing), 1 = does not exist.
# Any other failure (throttling, a permission gap) is fatal and shows its
# message: only CloudFormation's "Stack with id X does not exist" means absent.
describe_stack() {
  local err
  if err="$("${AWS[@]}" cloudformation describe-stacks --stack-name "$1" 2>&1 >/dev/null)"; then
    return 0
  fi
  if [[ "$err" == *"ValidationError"*"does not exist"* ]]; then
    return 1
  fi
  echo "error: cannot read stack $1: $err" >&2
  exit 1
}

stack_output() { # stack key
  "${AWS[@]}" cloudformation describe-stacks --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}

# Log groups to delete once their stacks are gone. Names come from the stack
# outputs, never from a broad pattern: the runtime id carries a random suffix,
# so its prefix matches only this runtime's endpoint groups.
LOG_GROUPS=()
LOG_PREFIXES=()

for suffix in runtime build; do
  name="${STACK}-${suffix}"
  if describe_stack "$name"; then
    if [[ "$suffix" == runtime ]]; then
      runtime_id="$(stack_output "$name" AgentRuntimeId)"
      if [[ -n "$runtime_id" && "$runtime_id" != "None" ]]; then
        LOG_PREFIXES+=("/aws/bedrock-agentcore/runtimes/${runtime_id}-")
      fi
    else
      project="$(stack_output "$name" BuildProjectName)"
      if [[ -n "$project" && "$project" != "None" ]]; then
        LOG_GROUPS+=("/aws/codebuild/${project}")
      fi
      # CloudFormation cannot delete a non-empty bucket; build contexts are
      # disposable, so empty it first.
      bucket="$(stack_output "$name" SourceBucketName)"
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

for prefix in ${LOG_PREFIXES[@]+"${LOG_PREFIXES[@]}"}; do
  while IFS= read -r group; do
    [[ -n "$group" && "$group" != "None" ]] && LOG_GROUPS+=("$group")
  done < <("${AWS[@]}" logs describe-log-groups --log-group-name-prefix "$prefix" \
             --query 'logGroups[].logGroupName' --output text | tr '\t' '\n')
done
for group in ${LOG_GROUPS[@]+"${LOG_GROUPS[@]}"}; do
  # Absent is fine (the service never wrote to it); anything else is reported.
  if err="$("${AWS[@]}" logs delete-log-group --log-group-name "$group" 2>&1)"; then
    echo "==> Deleted log group $group"
  elif [[ "$err" == *ResourceNotFoundException* ]]; then
    echo "==> Log group $group not found; skipping"
  else
    echo "error: cannot delete log group $group: $err" >&2
    exit 1
  fi
done
echo "Teardown complete."
