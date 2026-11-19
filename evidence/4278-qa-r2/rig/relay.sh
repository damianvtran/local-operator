# relay.sh <cfg> <port> [nocap] — PR-worktree relay as its own process-group leader; prints the pgid
. "$(dirname "$0")/env.sh"; W=/home/ec2-user/local-operator/.worktrees/bridge-card-phone-web
set -m
( cd $W && exec env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$S/home TERM=xterm-256color LANG=C.UTF-8 \
  LOCAL_OPERATOR_CONFIG_DIR=$S/cfg/$1 QA_SEQMAP=$S/cfg/$1.seqmap.json QA_WIRE_LOG=$S/cfg/$1.wire.jsonl \
  GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0 LO_MOBILE_NO_DIAL="${QA_NODIAL:-1}" LOP_MOBILE_PASSWORD="$(cat $S/pw)" QA_NO_CAP="${3:+1}" QA_EPOCH=$S/epoch QA_UNK=$S/unk_on QA_LATE="${QA_LATE:-}" \
  .venv/bin/python $S/rig/lop_fake.py mobile serve --port "$2" ) </dev/null > $S/relay-$1-$2.log 2>&1 &
echo $!
