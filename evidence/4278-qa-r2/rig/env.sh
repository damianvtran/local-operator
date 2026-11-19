# usage: qaenv <cfgname> cmd...   — isolated HOME/config, inherited LOP/LOCAL_OPERATOR/CMUX stripped
S=/home/ec2-user/.local-operator/sessions/eb00b020a26e/scratchpad/qa
qaenv() { local c=$1; shift; env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=$S/home TERM=xterm-256color LANG=C.UTF-8 \
  LOCAL_OPERATOR_CONFIG_DIR=$S/cfg/$c QA_SEQMAP=$S/cfg/$c.seqmap.json QA_WIRE_LOG=$S/cfg/$c.wire.jsonl \
  GIT_CONFIG_SYSTEM=/dev/null GIT_TERMINAL_PROMPT=0 LO_MOBILE_NO_DIAL=1 "$@"; }
