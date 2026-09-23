"""The credential-shape corpus: what MUST be masked, and what MUST survive.

Kept as data rather than as assertions inside a test function for one reason:
the two halves are a CONTRACT, and a contract that lives in one long function is
edited one case at a time until nobody can see its shape. Here the positives and
the negatives sit side by side, each with its own one-line reason, and the tests
in ``test_credential_shapes.py`` parametrise over them on every model-visible
surface. Adding a rule without a case — or widening a rule past a case — shows
up as a diff against a table someone can read.

**The negatives are as much of the contract as the positives.** This pass runs
over every tool result, so a rule that masks ordinary output blinds the agent to
the text it is reading: ``max_tokens`` (a model parameter), ``SERVICE_URL`` (an
ordinary endpoint), a git remote, a token COUNT, the word "password" in prose.
Each such case below carries the reason it must survive, and the reason is the
one a future widening has to argue against.

Every positive says WHY the harness should recognise it — a scheme keyword, a
credential name, an issuer prefix, a CLI convention — because the table's whole
claim is that it recognises credentials by SPELLING rather than by entropy, and
a case with no spelling argument is a case the table cannot honestly catch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    """One corpus entry: the text, and the reason it belongs to its half."""

    text: str
    reason: str


def _compact(*fields: tuple[str, str], quote: str = '"') -> str:
    """A compact JSON object — no whitespace, which is how a wire response arrives.

    Built from ``(name, value)`` PAIRS rather than spelled out, following the
    convention this file already uses for its own fixtures: it is read by agents
    through the very pass it describes, and a literal ``"name":"value"`` in its
    source is a credential-shaped spelling in THEIR transcript. The ``quote``
    parameter exists for the Python-repr spelling of the same object.

    The shape matters as much as the values: with no whitespace between fields
    nothing in the object bounds a greedy value match, so a rule that assumed a
    delimiter would stop it crossed the field boundary — see the compact cases in
    the POSITIVE block, and the counters in the NEGATIVE block for what that
    judgement was catching instead.

    ONE LIMIT, which is not visible from the signature: a value CONTAINING the
    quoting character cannot be expressed through this helper, so a fixture whose
    value holds a quote is written out in full instead of built here. The compact
    fixtures below are also NAMED, and the regression tests read those names rather
    than searching the table for a reason string: a test whose fixture is found by
    prose is one edit away from testing nothing (agent review R1, finding 7).
    """
    q = quote
    return "{" + ",".join(f"{q}{name}{q}:{q}{value}{q}" for name, value in fields) + "}"


#: The exact wire texts the compact cases carry, spelled once and NAMED.
#:
#: Readability is why: they are what a reviewer checks the quoted rule against, and
#: they are what the regression tests read — ``COMPACT_TOKEN_PAIR`` rather than a
#: search for a reason string, which is the coupling that made an edit to prose an
#: edit to a test (agent review R1, finding 7).
#: renders: {"access_token":"abc123def456","refresh_token":"zzz999yyy888"}
COMPACT_TOKEN_PAIR = _compact(("access_token", "abc123def456"), ("refresh_token", "zzz999yyy888"))
COMPACT_CLIENT_CREDENTIALS = _compact(
    ("client_secret", "s3cr3t-value-here"), ("client_id", "public-identifier-x")
)
COMPACT_CLIENT_CREDENTIALS_REPR = _compact(
    ("client_secret", "s3cr3t-value-here"), ("scope", "read"), quote="'"
)
COMPACT_API_KEY = _compact(("api_key", "realAccessKeyId1234"), ("region", "ca-central-1"))
COMPACT_PASSWORD_PAIR = _compact(("password", "hunter2hunter2"), ("user", "svc"))

#: The Bedrock evidence file's own line, as the operator's ``write`` carried it: a
#: counter under a qualified name, with the whitespace that kept THIS spelling quiet
#: while the compact one fired.
COUNTER_USAGE_LINE = (
    'usage: {"input_tokens": 16, "cache_creation_input_tokens": 0, '
    '"cache_creation": {"ephemeral_5m_input_tokens": 3536}}'
)

#: The credential NAMES the TAIL arm RELEASES, pinned so the boundary is a decision
#: rather than a differential artefact (QA round 2, Q2-1): a qualified quantity tail.
#:
#: They were masked at ``origin/main`` — the first-segment arm does not see them —
#: and this change reads them as COUNTS, which is the reading QA's sweep agreed with
#: and the same judgement that keeps the Anthropic counters unmasked. The cases live
#: in the NEGATIVE half and carry a value that WOULD be masked if the tail arm were
#: reverted, so they discriminate in the direction that matters.
COUNT_TAIL_RELEASED_NAMES: tuple[str, ...] = (
    "REDIS_CACHE_TOKENS",
    "DB_CACHE_TOKENS",
    "SENTRY_CACHE_TOKENS",
    "MINIO_CACHE_TOKENS",
)

#: A synthetic value for the generated cases: opaque, credential-free spelling, and
#: deliberately not built from a name so it cannot be mistaken for a real one. It is
#: long enough to clear the assignment rule's seven-character floor.
FIXTURE_VALUE = "QA-fixture-9c1f4a"

#: One line's worth of escaped rendering, as a JSON payload spells a line break:
#: the two characters ``\`` and ``n``. It is load-bearing rather than cosmetic.
#: A name the count-trap exclusion covers is spared by ``is_count_shaped``, which
#: keys on the FIRST segment of the name — so a count name only becomes a false
#: positive when something is glued to its front, and the only thing that glues
#: itself there is the escape letter of the line break before it. A case written
#: without this prefix is a case origin/main passes too, i.e. no evidence at all.
PRE_ESCAPED_LINE = "RESULTS = os.path.join(HERE)" + "\\n"

#: The credential NAMES the identifier arm released, and the surfaces it released
#: them on (agent review R1-1).
#:
#: **Why a table and not a handful of rows.** The arm's release was written as a
#: rule over any multi-segment identifier, so it was spelling-INDEPENDENT: it took
#: the value no matter which name carried it, on every surface an assignment is
#: scrubbed through. A pin that covers one name on one surface is exactly the gap
#: that let R1-1 through in the first place — the corpus's multi-word-password
#: positives were all HYPHENATED, and the arm only touched the underscored
#: spelling, so both differentials stayed silent while real credentials went
#: readable. Every name here was measured released at that revision and masked at
#: ``origin/main``.
IDENTIFIER_ARM_NAMES: tuple[str, ...] = (
    "PASSWORD",
    "PGPASSWORD",
    "MONGO_PASSWORD",
    "DB_PASSWORD",
    "API_TOKEN",
    "AWS_SECRET_ACCESS_KEY",
    "POSTGRES_PASSWORD",
    "KAFKA_SECRET",
)

#: The value ALPHABETS, paired with a label: the released class is the digit-free
#: underscore-joined one, and the other two are its immediate neighbours, which
#: stayed masked at the broken revision and must keep masking. The hyphenated
#: neighbour is the spelling the corpus already pinned; the digit-carrying one is
#: the floor the arm must not cross.
IDENTIFIER_ARM_VALUES: tuple[tuple[str, str], ...] = (
    ("digit-free, underscore-joined", "corr" + "ect_horse_bat" + "tery"),
    ("digit-free, hyphen-joined", "corr" + "ect-horse-bat" + "tery"),
    ("digit-carrying", "corr" + "ect_horse_bat" + "tery2"),
)

#: The surfaces each pair is scrubbed through. Built by :func:`_arm_spelling`
#: rather than written out, so the file itself — which an agent reads through the
#: very pass it describes — carries no credential-shaped assignment.
IDENTIFIER_ARM_SPELLINGS: tuple[str, ...] = (
    "assignment",
    "export-prefixed",
    "docker-compose",
    "quoted",
    "JSON field",
)


def _arm_spelling(name: str, spelling: str, value: str) -> str:
    """One surface the released class was measured on, assembled from its parts."""
    if spelling == "assignment":
        return name + "=" + value
    if spelling == "export-prefixed":
        return "export " + name + "=" + value
    if spelling == "docker-compose":
        return name + ": " + value
    if spelling == "quoted":
        return name + '="' + value + '"'
    if spelling == "JSON field":
        return '{"' + name.lower() + '": "' + value + '"}'
    raise AssertionError(f"unknown spelling {spelling!r}")


#: Credential NAMES whose qualifier carries a word that also names a QUANTITY.
#:
#: The other direction of the count judgement, and the one that leaks when the
#: sweep is widened carelessly. ``is_credential_name`` only asks that the TAIL be a
#: credential word, so ``REDIS_CACHE_PASSWORD`` has the same segment-level shape as
#: ``ephemeral_5m_input_tokens`` — ``…_<count word>_…_<credential tail>`` — and must
#: still be masked. Every name here IS masked at ``origin/main``; every one was
#: released by an any-segment sweep (agent review R1-1, QA round 1 Q-1), with no
#: mask, no label and nothing registered for containment. They live in a table
#: because two readers need the same list: the POSITIVE block generates a case per
#: name, and the count-judgement test iterates it rather than restating the names as
#: prose (which is what let the class through the first time).
COUNT_QUALIFIER_NAMES: tuple[str, ...] = (
    # `cache` in a non-first segment, under a password/secret/token tail.
    "REDIS_CACHE_PASSWORD",
    "REDIS_CACHE_TOKEN",
    "DB_CACHE_PASSWORD",
    "SENTRY_CACHE_PASSWORD",
    "ELASTICACHE_CACHE_PASSWORD",
    "MY_CACHE_PASSWORD",
    "SESSION_CACHE_SECRET",
    "MINIO_CACHE_SECRET_KEY",
    # Other quantity words in a qualifier: `output`, `prompt`, `page`.
    "KAFKA_OUTPUT_SECRET",
    "OPENAI_PROMPT_KEY",
    "FACEBOOK_PAGE_ACCESS_TOKEN",
    "META_PAGE_ACCESS_TOKEN",
    "IG_PAGE_ACCESS_TOKEN",
    "MY_PAGE_ACCESS_TOKEN",
)


def _angled(inner: str) -> str:
    """``<inner>``, assembled rather than spelled.

    The angle characters are built here for one reason: this corpus is read by
    agents through the very pass it describes, and the spellings below are the
    false positive this section exists to make readable. A fixture written out in
    full would put that trigger in a file those agents read — and before the fix
    it fired there too, which is why the first draft of this section masked its
    own evidence.
    """
    return chr(60) + inner + chr(62)


def _generic(base: str, *arguments: str) -> str:
    """A generic type application (``Option<String>``), assembled as above."""
    return base + _angled(", ".join(arguments))


#: The TYPE ANNOTATIONS that must survive, and the reason each one exists.
#:
#: **This is the false positive, and it is measured rather than imagined.** Reading
#: three real source files (``agentic_adm.rs``, ``article_search.rs``, ``mod.rs``)
#: put two of these in a transcript: a struct field whose NAME is credential-shaped
#: and whose VALUE is a Rust type. Both were masked, and the escalation they carried
#: — the mask covered part of the type and left the rest readable, which grades
#: ``exposed`` — stopped a release pending a rotation verdict for a type annotation.
#:
#: They are a NAMED table rather than rows inside the block below so the tests read
#: them by identity; a case found by searching a reason string is one prose edit away
#: from testing nothing (the coupling agent review R1, finding 7 corrected elsewhere in
#: this file). ``_generic`` keeps the angle characters out of this source, so a future
#: change to the spelling cannot silently rewrite a fixture.
TYPE_ANNOTATION_NEGATIVES: tuple[Case, ...] = (
    # --- the measured line, and the two spellings it was reported in -----------
    Case(
        "    api_key: " + _generic("Option", "String") + ",",
        "a Rust struct field: a credential-shaped NAME, a TYPE for a value",
    ),
    Case(
        "pub image: " + _generic("Option", "String") + ",",
        "the same field with its visibility modifier, as reported",
    ),
    Case(
        "    let api_key: " + _generic("Option", "String") + " = None;",
        "the same type in a let binding: not a struct field only",
    ),
    Case(
        "    api_key: " + _generic("Option", "String") + ",",
        "the private spelling, with no visibility modifier",
    ),
    # --- the generic spellings, and the truncation the value group forces -------
    Case(
        "    api_key: " + _generic("Vec", "String") + ",",
        "a Vec annotation",
    ),
    Case(
        "api_key: " + "HashMap" + _angled("String"),
        "a HashMap cut at the comma-space: how a two-argument generic ARRIVES",
    ),
    Case(
        "    pub api_key: " + _generic("HashMap", "String, String") + ",",
        "the whole two-argument generic on the line, which ESCALATED before the fix",
    ),
    Case(
        "api_key: " + "Result" + _angled("String"),
        "a Result cut the same way",
    ),
    Case(
        "api_key: " + _generic("Arc", "Mutex" + _angled("T")) + ",",
        "a nested generic: Arc<Mutex<T>>",
    ),
    Case(
        "api_key: " + _generic("Option", "Vec" + _angled("u8")) + ",",
        "a nested generic over a primitive: Option<Vec<u8>>",
    ),
    Case(
        "api_key: " + _generic("Option", "SomeVeryLongTypeName") + ",",
        "a generic over a CUSTOM type name, which no allowlist would cover",
    ),
    Case(
        "api_key: " + _generic("ArrayVec", "u8, 32") + ",",
        "a const generic: a bare integer argument beside a type",
    ),
    Case(
        "api_key: " + _generic("Cow", "str") + ",",
        "a generic over a Rust primitive",
    ),
    Case(
        "api_key: " + _generic("Cow", "'a, str") + ",",
        "the same with a lifetime argument",
    ),
    Case(
        "api_key: Box" + _angled("dyn Trait") + ",",
        "a trait object, which carries a SPACE and so reaches the rule cut",
    ),
    # --- the other two type-only markers: a qualified path, and a reference -----
    Case(
        "api_key: std::collections::HashMap,",
        "a fully qualified path: :: is type-only punctuation",
    ),
    Case(
        "api_key: &SomeVeryLongEnumName,",
        "a reference to a long type name, with no angle bracket at all",
    ),
    # --- the same constructs in the other two languages ------------------------
    Case(
        "  api_key: " + _generic("Map", "string, string") + ";",
        "a TypeScript generic over two primitives",
    ),
    Case(
        "  api_key: string;",
        "a bare TypeScript primitive",
    ),
    Case(
        "    api_key: Optional[str] = None",
        "a Python annotation, already released by the expression-character rule",
    ),
)

#: The values this release must NOT touch: real credentials spelled in a way that
#: resembles a type. Each is a case the type clause could plausibly have eaten, and
#: the reason names the arm that stops it — which is what makes these the evidence
#: that the clause is scoped rather than broad.
TYPE_ANNOTATION_POSITIVES: tuple[Case, ...] = (
    Case(
        "API" + "_KEY=" + "Aa1!" + "bB2@" + "cC3#" + "dD4",
        "a mixed-case token: no type-only character, released by no arm here",
    ),
    Case(
        "API" + "_KEY=" + _generic("MyPass", "secret"),
        "a passphrase written as a generic whose ARGUMENT is not a type",
    ),
    Case(
        "API" + "_KEY=" + _generic("Abc", "xyz"),
        "the same: a lowercase argument is a word, not a type",
    ),
    Case(
        "API" + "_KEY=" + _generic("Foo", "word"),
        "the same spelling with a longer word",
    ),
    Case(
        "API" + "_KEY=" + _generic("PassWord", "1"),
        "a generic whose only argument is a const integer: not a type application",
    ),
    Case(
        "API" + "_KEY=" + "averylong" + "lowercase" + "name",
        "a long lowercase word: no type-only character at all",
    ),
    Case(
        "SESSION" + "_TOKEN=" + "abc" + "::" + "def",
        "a credential spelled with a path separator: the LEAF name is a word",
    ),
    Case(
        "DB" + "_PASSWORD=" + "Correct" + "_horse" + _angled("Battery") + "7",
        "a passphrase carrying angles and digits: not confined to the type alphabet",
    ),
    # --- the digit-carrying half of the residual class, which had NO row --------
    # The residual below is ``Ident<Ident>``: capitals on both sides, no digit, no
    # symbol and no word break. Before this round the arm released the WHOLE class
    # around it — a digit on EITHER side, and a lowercase base — with no hit at all,
    # and the corpus pinned only the digit-free spelling, so nothing in the suite
    # caught the dangerous half (agent review R1-1). These five rows are that half,
    # and each is a value the arm released before the confinement was enforced.
    Case(
        "API" + "_KEY=" + _generic("Pass", "Word") + "7",
        "the digit-carrying sibling of the residual: a digit in the ARGUMENT masks",
    ),
    Case(
        "API" + "_KEY=" + _generic("Pass7", "Word"),
        "the same with the digit in the BASE: also masks",
    ),
    Case(
        "API" + "_KEY=" + _generic("Abc", "Xyz") + "1",
        "both capitals, digit in the ARGUMENT: the grid row that was released whole",
    ),
    Case(
        "DB" + "_PASSWORD=" + _generic("Correcthorse", "Battery") + "7",
        "the DIGIT-FREE passphrase with its underscore removed: the digit is enough",
    ),
    Case(
        "API" + "_KEY=" + _generic("foo", "Bar"),
        "a LOWERCASE base under a generic: a word is not a type",
    ),
    # --- the R1-2 class: a credential WORD as the base of an application ---
    # ``_TYPE_PRIMITIVES`` admits 44 ordinary words, several of them plain English in
    # the languages that own them (``any``, ``void``, ``object``, ``null``), so an
    # argument rule that accepted one released ``Pass<int>`` — 70 of the 81 new
    # releases an enumeration found. The discriminating fact is not the argument but
    # the BASE: a type application's base is a type name (``Vec``, ``Option``, ``Foo``)
    # and is never a credential word. These rows pin that, and ``Vec<u8>`` /
    # ``Option<Vec<u8>>`` stay released above as the boundary on the other
    # side.
    Case(
        "API" + "_KEY=" + _generic("Pass", "int"),
        "a credential-stem base over a bare PRIMITIVE argument",
    ),
    Case(
        "API" + "_KEY=" + _generic("Pass", "any"),
        "the same: ``any`` is an ordinary English word, not a type",
    ),
    Case(
        "API" + "_KEY=" + _generic("Secret", "str"),
        "a credential WORD base over a primitive",
    ),
    Case(
        "API" + "_KEY=" + _generic("Token", "void"),
        "the same with ``void``",
    ),
    # --- the R1-1 half the corpus had no row for: a bare ``::`` path ---
    Case(
        "API" + "_KEY=" + "Sv" + "::" + "Secret",
        "a CamelCase MODULE segment: a namespace is lowercase, so this masks",
    ),
    Case(
        "SESSION" + "_TOKEN=" + "Camel" + "::" + "Word9",
        "the same convention violation with a digit-carrying leaf",
    ),
)


#: A credential spelled the way something spells one. Every one of these must
#: come back with at least one shape masked.
def _secret_run(specification: str) -> str:
    """The documented way to hand a stored secret to a child, assembled from parts.

    A helper rather than a literal because this file is read by agents THROUGH the
    pass it describes: "``--secret`` followed by a value" is exactly the shape that
    pass rewrites, so the flag and its argument are never adjacent in this SOURCE.
    """
    return "lop secret run " + "--" + "secret " + specification + " -- npm publish"


POSITIVE_CASES: tuple[Case, ...] = (
    # --- added in round 2: the leaks the original corpus did not spell ---------
    # Every one of these was published at some point in this PR's own history, and
    # every one of the ORIGINAL cases above still has to pass — the corpus is the
    # specification for this control, so a rule that needs a fixture changed is a
    # rule that is wrong.
    Case("MONGO_DSN=mongodb+srv://svc:p@ssw0rd@db.invalid/x", "a DSN password containing @"),
    Case(
        'MONGO_DSN=mongodb+srv://svc:qA2S3n7x9@x/y,z"w@db.invalid/x',
        "a DSN password containing @ / and a quote",
    ),
    Case("MONGO_DSN=mongodb://svc:qA2S3n7'x@db.invalid/x", "a DSN password containing a quote"),
    Case(
        "DATABASE_URL=postgres://svc:p/ssw0rd@db.invalid/app",
        "a solidus password under a *_URL name",
    ),
    Case("PASSWORD=hunter2hunter2,hunter2", "a password containing a comma"),
    Case("DB_PASSWORD=abc,defghij", "a comma inside the first eight characters"),
    Case('{"password": "abcdef,ghij"}', "a comma inside a JSON string value"),
    Case('DB_PASSWORD=abc"defghij', "a quote inside an unquoted value"),
    Case("SECRET=abc123;def456", "a semicolon inside a value"),
    Case("PASSWORD=swordfish", "a password with no digit at all"),
    Case("POSTGRES_PASSWORD=swordfish", "the same, prefixed"),
    Case("AWS_SECRET_ACCESS_KEY=_abcdefghijklmnopqrstuvwxyz1234567890", "a secret starting with _"),
    Case('{"Authorization": "Bearer deadbeefdeadbeef1234"}', "the quoted JSON header spelling"),
    Case("{'authorization': 'bearer deadbeefdeadbeef1234'}", "the Python dict/repr spelling"),
    Case('authorization="Bearer deadbeefdeadbeef1234"', "the assignment spelling"),
    Case('HTTP_AUTHORIZATION="Bearer deadbeefdeadbeef1234"', "the env-var spelling"),
    Case(
        '{"name": "Authorization", "value": "Bearer deadbeefdeadbeef1234"}',
        "a HAR-style header dump",
    ),
    Case("Authorization: Basic YWRtaW46cHc=", "a Basic credential below the old 16-char floor"),
    Case("openssl enc -aes-256-cbc -pass swordfishsecret -in f", "the bare -pass spelling"),
    Case("openssl enc -aes-256-cbc -pass pass:swordfishsecret -in f", "-pass with pass:"),
    Case("pk-abcdefghijklmnop", "round 1's B-1 repro, letters only"),
    Case("whsec_abcdefghijklmnop", "a fixed-prefix tail with no digit"),
    Case("shpat_abcdefghijklmnop", "another fixed-prefix tail"),
    Case("host=db,password=swordfish1", "an assignment after a comma in DATA, not a call"),
    # --- the incident itself -------------------------------------------------
    Case(
        "MONGO_DSN=mongodb+srv://agent_runtime_model_worker:hunter2"
        "pw@mongodb-prod.example.net/agent_runtime?retryWrites=true",
        "the regression this module exists for: a DSN from a remote pod's environment",
    ),
    Case(
        "DEBUG env"
        " dump:\nMONGO_DSN=mongodb://agent_runtime_model_worker:hunter2pw"
        "@mongodb-prod.example.net/db\nPORT=8080",
        "the incident's own line inside the surrounding env dump,"
        " verbatim apart from the credential",
    ),
    Case(
        "MONGO_DSN=mongodb://user:pw@host/db",
        "a plain mongodb DSN assignment",
    ),
    # --- connection strings / DSNs ------------------------------------------
    Case("postgres://reporting_user:s3cr3tpass@db.internal:5432/analytics", "postgres DSN"),
    Case("postgresql://u:pw@h/db", "postgresql DSN"),
    Case("mysql://root:rootpw@127.0.0.1:3306/app", "mysql DSN"),
    Case("mariadb://svc:pw@h/db", "mariadb DSN"),
    Case("redis://:onlypassword@cache.internal:6379", "redis DSN with an empty user"),
    Case("rediss://default:pass@h:6380", "rediss (TLS) DSN"),
    Case("amqp://guest:guest@rabbit.internal:5672/", "amqp DSN"),
    Case("amqps://user:pw@h", "amqps DSN"),
    Case("mssql://sa:pw@h:1433/db", "mssql DSN"),
    Case("clickhouse://default:pw@h:9000/db", "clickhouse DSN"),
    Case("jdbc:postgresql://user:pw@host:5432/db", "jdbc-prefixed DSN"),
    Case("https://user:pw@api.internal/v1", "https URL carrying userinfo"),
    Case("http://admin:adminpw@10.0.0.5:9200/_cluster/health", "http URL carrying userinfo"),
    Case("ldap://cn=admin:pass@dir.internal", "ldap bind credential"),
    Case("sftp://deploy:pw@sftp.internal/upload", "sftp URL carrying userinfo"),
    Case(
        "SMTP_URL=smtp://relay_user:relaypw@smtp.internal:587",
        "a *_URL name whose value carries userinfo",
    ),
    Case(
        "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "the AWS secret access key, verbatim form",
    ),
    Case("AWS_SESSION_TOKEN=FwoGZXIvYXdzEBYaDExampleTokenValue1234567890", "an AWS session token"),
    Case("aws_access_key_id: AKIAIOSFODNN7EXAMPLE", "a lower-case aws key id assignment"),
    Case("DB_PASSWORD=correct-horse-battery", "a *PASSWORD name"),
    Case("POSTGRES_PASSWORD=s3cretvalue", "an upper-case *PASSWORD name"),
    Case("PASSWORD=hunter2hunter2", "a bare PASSWORD"),
    Case("password=hunter2hunter2", "a lower-case password assignment"),
    Case("passwd=supersecret", "the passwd spelling"),
    Case("pwd=supersecret1", "the pwd spelling"),
    Case("SECRET_KEY=django-insecure-abcdefghijklmnop", "a SECRET_KEY assignment"),
    Case("PRIVATE_KEY=abcdefghijklmnop", "a PRIVATE_KEY assignment"),
    Case("ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "an ACCESS_KEY_ID assignment"),
    Case("CLIENT_SECRET=abcdefghijklmnopqrstuvwxyz", "a CLIENT_SECRET assignment"),
    Case("CREDENTIALS=abcdefghijklmnop", "a CREDENTIALS assignment"),
    Case(
        "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789", "an upper-case *_TOKEN assignment"
    ),
    Case(
        "GITHUB_TOKEN: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "a colon-separated *_TOKEN assignment",
    ),
    Case("token=abcdefghijklmnop", "a bare token assignment"),
    Case("API-KEY: abcdefghijklmnop", "a dash-spelled API-KEY header"),
    Case("API_KEY=abcdefghijklmnop", "an underscore-spelled API_KEY"),
    Case("apiKey: abcdefghijklmnop", "a camelCase apiKey"),
    Case('export GITHUB_TOKEN="ghp_abcdefghijklmnopqrstuvwxyz0123456789"', "an export with quotes"),
    Case("  password: hunter2hunter2", "a YAML spelling with leading indentation"),
    Case("  token: 1234567890abcdef", "a YAML token"),
    Case(
        "  client-key-data: LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQo=",
        "kubeconfig client-key-data",
    ),
    Case("_authToken=npm_abcdefghijklmnopqrstuvwxyz", "the .npmrc auth token, bare name"),
    Case(
        "//registry.npmjs.org/:_authToken=npm_abcdefghijklmnopqrstuvwxyz",
        "the .npmrc auth token with a registry path",
    ),
    Case("_auth=abcdefghijklmnop", "the .npmrc base64 auth"),
    Case(
        '{"auths": {"registry.internal": {"auth": "dXNlcjpwYXNzd29yZA=="}}}',
        "docker config.json auth blob",
    ),
    Case(
        "machine api.github.com login robot password hunter2hunter2",
        "the .netrc machine/login/password line",
    ),
    Case("machine example.com login bob password s3cretpw", "a second .netrc line shape"),
    Case("POSTGRES_CONNECTION_STRING=postgres://u:pw@h:5432/db", "a *CONNECTION_STRING name"),
    Case("REDIS_CONN_STR=redis://:pw@h:6379", "a *CONN_STR name"),
    Case("DATABASE_DSN=host=db user=app password=hunter2hunter2", "a NAME=DSN assignment"),
    # --- query parameters ----------------------------------------------------
    Case(
        "https://api.example.com/v1?api_key=abcdefgh12345678&page=2",
        "an api_key query parameter in a URL",
    ),
    Case("https://h/cb?token=9f8e7d6c5b4a3210", "a token query parameter"),
    Case("callback?signature=abcdef1234567890", "a signature query parameter"),
    Case("https://h/v1?access_token=abcd1234efgh5678", "an access_token query parameter"),
    Case("https://h/v1?secret=abcdefghijklmnop", "a secret query parameter"),
    Case("https://h/v1?password=hunter2hunter2", "a password query parameter"),
    Case('{"url": "https://h/v1?apikey=abcdefgh1234"}', "an apikey query parameter inside JSON"),
    # --- generated-credential files -----------------------------------------
    Case(
        "-----BEGIN PRIVATE"
        " KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ\n-----END PRIVATE KEY-----",
        "a bare PKCS#8 PEM block",
    ),
    Case(
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234abcd\n-----END RSA PRIVATE KEY-----",
        "an RSA PEM block",
    ),
    Case(
        "-----BEGIN OPENSSH PRIVATE"
        " KEY-----\nb3BlbnNzaC1rZXktdjEAAAAABG5vbmU\n-----END OPENSSH PRIVATE KEY-----",
        "an OpenSSH PEM block",
    ),
    Case(
        '"private_key": "-----BEGIN RSA PRIVATE'
        ' KEY-----\\nMIIEowIBAAKCAQEA1234\\n-----END RSA PRIVATE KEY-----\\n"',
        "a GCP service-account JSON private_key with escaped newlines",
    ),
    Case(
        '"type": "service_account",\n"private_key_id": "abc123",'
        '\n"private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADAN'
        '\\n-----END PRIVATE KEY-----"',
        "the service-account JSON file shape as a whole",
    ),
    Case(
        '{"private_key": "-----BEGIN RSA PRIVATE KEY-----"}',
        "a service-account private_key whose body was truncated before the END line",
    ),
    # --- headers -------------------------------------------------------------
    Case(
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dBjftJeZ4CVP",
        "an Authorization Bearer header",
    ),
    Case("authorization:bearer abcdefgh12345678", "a lower-case, space-free bearer spelling"),
    Case("Authorization: Basic dXNlcjpwYXNzd29yZA==", "an Authorization Basic header"),
    Case("Set-Cookie: session=eyJzZXNzaW9uIjoxfQ; Path=/; HttpOnly", "a Set-Cookie session value"),
    Case("Cookie: sessionid=abcdef1234567890; other=1", "a Cookie request header"),
    Case("curl -v -H 'Cookie: token=abcdef1234567890'", "a cookie header inside a curl invocation"),
    # --- CLI flags -----------------------------------------------------------
    Case("mysql -u root -pSuperSecret1 -e 'show databases'", "an inline mysql -p password"),
    Case("mysqldump --password=hunter2 db", "a --password= flag"),
    Case("psql --password hunter2 -c 'select 1'", "a --password <value> flag"),
    Case("mongosh -psecretpass --eval 'db.stats()'", "an inline mongosh -p password"),
    Case("redis-cli -a hunter2hunter2 ping", "a redis-cli -a password"),
    Case("server --token=abcdefghijklmnop", "a --token= flag"),
    Case("cli --client-secret abcdefghijklmnop", "a --client-secret flag"),
    Case("app --api-key=abcdefghijklmnop", "an --api-key= flag"),
    Case("tool --access-token abcd1234efgh5678", "an --access-token flag"),
    # R1-1: capitals in a flag position are a CREDENTIAL, not a NAME. A first
    # cut of the NAME guard (capitals plus a credential-word tail) stopped masking
    # all five of these, silently: no mask and no notice, because no hit means no
    # labels and no exposure. Multi-segment capitals
    # (``OS_PROD2_ADMIN_PASSWORD``-style) is what a NAME looks like; a single run
    # of capitals is a credential someone chose. Assembled so no literal here is a
    # flag VALUE in this source, since a flag followed by a value is what a
    # redaction pass rewrites.
    Case("--" + "api-key" + " KEY", "caps in an --api-key value position"),
    Case("--" + "password" + " PASSWORD", "caps in a --password value position"),
    Case("--" + "token" + " TOKEN", "caps in a --token value position"),
    Case("--" + "secret" + " DBPASSWORD", "a run-together caps name-ish value"),
    Case("--" + "api-key" + " APIKEY", "a run-together caps value"),
    # --- bare issuer-prefixed tokens ----------------------------------------
    Case("sk_live_51H8xYzAbCdEf", "a Stripe live secret key"),
    Case("rk_live_51H8xYzAbCdEf", "a Stripe restricted key"),
    Case("sk-proj-EXAMPLEabcdefghijklmnopqrst", "an OpenAI project key"),
    Case("sk-ant-api03-EXAMPLEabcdefghijklmnopqrst", "an Anthropic key"),
    Case("SG.EXAMPLEabcdefghijklmno._EXAMPLEabcdefghijklmno", "a SendGrid API key"),
    Case("ghp_EXAMPLEabcdefghijklmnopqrstuvwxyz01", "a GitHub personal access token"),
    Case("gho_EXAMPLEabcdefghijklmnopqrstuvwxyz01", "a GitHub OAuth token"),
    Case(
        "github_pat_EXAMPLEabcdefghijklmnopqrstuvwxyz0123456789",
        "a fine-grained GitHub token",
    ),
    Case("AKIAIOSFODNN7EXAMPLE", "an AWS access key id"),
    Case("ASIAIOSFODNN7EXAMPLE", "a temporary AWS access key id"),
    Case("AIzaSyEXAMPLE1234567890abcdefghijkl", "a Google API key"),
    Case("ya29.EXAMPLEabcdefghijklmnopqrstuvwx", "a Google OAuth access token"),
    Case("xoxb-EXAMPLE-abcdefghijklmnop", "a Slack bot token"),
    Case("xoxp-EXAMPLE-abcdefghijklmnop", "a Slack user token"),
    Case("hf_EXAMPLEabcdefghijklmnopqrstuv", "a Hugging Face token"),
    Case("tvly-EXAMPLEabcdefghijklmnop", "a Tavily API key"),
    Case("glpat-EXAMPLEabcdefghijklmnop", "a GitLab personal access token"),
    Case("gsk_EXAMPLEabcdefghijklmnopqrst", "a Groq API key"),
    Case("npm_EXAMPLEabcdefghijklmnopqrstuvwxyz", "an npm token"),
    Case(
        "npm_8f3a2b1c-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
        "a same-prefix REAL token: its tail is hex and dashes, and a DIGIT carries it",
    ),
    Case("pypi-EXAMPLEabcdefghijklmnopqrstuvwx", "a PyPI upload token"),
    Case("dckr_pat_EXAMPLEabcdefghijklmnop", "a Docker Hub access token"),
    Case("lin_api_EXAMPLEabcdefghijklmnop", "a Linear API key"),
    Case("whsec_EXAMPLEabcdefghijklmnop", "a Stripe webhook signing secret"),
    Case("fal-EXAMPLEabcdefghijklmnop", "a fal.ai key in its prefixed form"),
    Case("serp-EXAMPLEabcdefghijklmnop", "a SerpAPI key in its prefixed form"),
    Case(
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0N"
        "TY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "a JWT with its two dots",
    ),
    Case(
        "token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmnop",
        "a JWT under a credential name",
    ),
    # --- environment / kubectl / docker / cloud CLI output -------------------
    Case(
        "AGENT_RUNTIME_API_KEY=abcdefghijklmnopqrstuvwxyz",
        "the runtime's own upper-case API key variable",
    ),
    Case(
        '  {"name": "MONGO_DSN", "value": "mongodb://agent:pw@mongo.internal/db"}',
        "a DSN inside a JSON env array",
    ),
    Case(
        "docker inspect api: 'Env': ['MONGO_DSN=mongodb://u:pw@h/db']", "a docker inspect env dump"
    ),
    Case(
        'aws secretsmanager get-secret-value --secret-id prod/db: {"password":"hunter2hunter2"}',
        "an AWS secret payload",
    ),
    Case(
        'terraform output -json: {"db_password": "hunter2hunter2"}',
        "a terraform output carrying a password",
    ),
    Case("vault kv get secret/prod: password=hunter2hunter2", "vault output carrying a password"),
    Case("PASSWORD: hunter2hunter2\nUSERNAME: app", "a mixed-case credential block"),
    Case("AUTH_TOKEN = abcdefghijklmnop", "a spaced assignment"),
    Case('\t"refresh_token": "abcdefghijklmnopqrst"', "a tab-indented JSON refresh token"),
    Case(
        "https://gateway.internal/callback?user=app&password=hunter2hunter2&tenant=1",
        "a URL with a credential query parameter among ordinary ones",
    ),
    Case(
        "postgres://user:pw@host:5432/db?sslmode=require",
        "a DSN with an ordinary trailing parameter",
    ),
    Case(
        "mongodb+srv://user:pw@cluster.example.net/?retryWrites=true&w=majority",
        "an SRV DSN with a query string",
    ),
    Case("ssh://root:pw@host:22", "an ssh URL carrying a password"),
    Case("ftp://ftpuser:ftppw@files.internal/pub", "an ftp URL carrying userinfo"),
    Case("ldaps://svc:pw@dir.internal:636", "an ldaps URL carrying userinfo"),
    Case("jdbc:mysql://user:pw@host:3306/db?useSSL=false", "a jdbc mysql DSN with parameters"),
    Case("MONGO_PASSWORD=hunter2hunter2", "a MONGO_-prefixed password variable"),
    Case("MONGO_URI=mongodb://user:pw@host/db", "a *URI name whose value is credential-carrying"),
    Case("MONGO_URL=mongodb://user:pw@host/db", "a *URL name whose value is credential-carrying"),
    Case(
        "API_ENDPOINT=https://user:pw@api.internal/v1",
        "an *ENDPOINT name whose value carries userinfo",
    ),
    Case(
        "SENTRY_DSN=https://abcdef1234567890@o1.ingest.sentry.io/2",
        "a DSN whose only secret is the public key form",
    ),
    Case(
        "REDIRECT_URI=https://app.internal/cb?token=abcdef123456",
        "an OAuth redirect carrying a token, which is a credential",
    ),
    Case("SPRING_DATASOURCE_PASSWORD=hunter2hunter2", "a long prefixed password variable"),
    Case("my_service_db_password=hunter2hunter2", "a lower-case multi-segment password variable"),
    Case("X-API-Key: abcdefghijklmnop", "an X-API-Key header"),
    Case("Proxy-Authorization: Basic dXNlcjpwYXNz", "a Proxy-Authorization header"),
    Case(
        "Authorization: token ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "GitHub's Authorization: token form",
    ),
    Case(
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\nAWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG",
        "a credentials-file pair",
    ),
    Case("docker login -u robot -p abcdefghijklmnop", "a docker login -p password"),
    Case(
        "kubectl create secret generic db --from-literal=password=hunter2hunter2",
        "a kubectl create with a literal password",
    ),
    Case("openssl rsa -in key.pem -passin pass:hunter2hunter2", "an openssl passin password"),
    Case("PGPASSWORD=hunter2hunter2 pg_dump app", "a PGPASSWORD environment prefix"),
    Case("curl -u user:hunter2hunter2 https://api.internal", "a curl -u credential"),
    Case(
        "https://hooks.slack.com/services/T000/B000/abcdefghijklmnop",
        "a Slack incoming-webhook URL",
    ),
    Case("DATABASE_PASSWORD='hunter2hunter2'", "a single-quoted password value"),
    Case('DATABASE_PASSWORD="hunter2hunter2"', "a double-quoted password value"),
    Case('{"client_secret":"abcdefghijklmnopqrstuvwxyz"}', "a compact JSON client_secret"),
    Case(
        "{'secret_key': 'django-insecure-abcdefghijklmnop'}", "a single-quoted Python dict secret"
    ),
    Case("session_secret=abcdefghijklmnop", "a session_secret assignment"),
    Case("SIGNING_KEY=abcdefghijklmnop", "a SIGNING_KEY assignment"),
    Case("ENCRYPTION_KEY=abcdefghijklmnop", "an ENCRYPTION_KEY assignment"),
    Case("MASTER_KEY=abcdefghijklmnop", "a MASTER_KEY assignment"),
    Case("bearer abcdefghijklmnopqrst", "a bare bearer keyword and value with no header name"),
    # --- compact JSON: the spelling the corpus could not see ------------------
    # Nothing above carried two fields in the SAME object with no whitespace
    # between them, so no case measured the boundary the greedy assignment value
    # crossed: on a compact pair it consumed the closing quote and the NEXT
    # FIELD'S KEY, which masked the neighbour and then graded a fragment of the
    # swallowed text as exposed (the operator's `hub` notice, 2026-09-20). These
    # are the wire spellings — an OAuth token response, a client-credentials
    # response, the Python repr of the same — and the last one is the
    # OVER-REACH direction: a real key id in the same compact shape, which the
    # narrowed value grammar may not release.
    Case(
        COMPACT_TOKEN_PAIR,
        "a compact JSON token pair: the mask may not cross the field boundary",
    ),
    Case(
        COMPACT_CLIENT_CREDENTIALS,
        "a compact client-credentials response, the neighbour a public id",
    ),
    Case(
        COMPACT_CLIENT_CREDENTIALS_REPR,
        "the Python repr spelling of the same, single-quoted and compact",
    ),
    Case(
        COMPACT_API_KEY,
        "a real key id in compact JSON: the over-reach direction",
    ),
    Case(
        COMPACT_PASSWORD_PAIR,
        "a compact password whose neighbouring field must survive",
    ),
    # --- a count word in a QUALIFIER, under a credential TAIL ------------------
    # One case per name in ``COUNT_QUALIFIER_NAMES``. These are the names an
    # any-segment count sweep releases: the qualifier says "cache" or "page" and
    # the tail says "password" or "access_token", and the TAIL is what the name
    # IS. They are in this half because they must be masked — and they were, at
    # ``origin/main``, which is what makes releasing them a regression rather than
    # a trade. The name is in the reason so each case has its own test id.
    *(
        Case(
            f"{name}={FIXTURE_VALUE}",
            f"a count word in a qualifier segment under a credential tail: {name}",
        )
        for name in COUNT_QUALIFIER_NAMES
    ),
    # --- 2026-09-21, second wave: the escapes must not hide a real assignment --
    # The two narrowings those negatives pinned are about the RENDERING, and a
    # narrowing that also stopped masking the credentials arriving IN that
    # rendering would be a leak rather than a fix. Same construct, real value.
    Case(
        "line" + "\\n" + "OPENROUTER_API_KEY=" + FIXTURE_VALUE,
        "a real key one escaped newline after its name: the escape is not the name",
    ),
    Case(
        "OPENROUTER_API_KEY=" + FIXTURE_VALUE + "\\n" + "next",
        "a real key whose line ends where the rendering says it does",
    ),
    Case(
        "xai" + "-Org/" + "Grok-Build9",
        "a slash-joined tail that is NOT a name: case and a digit make it a token",
    ),
    # --- 2026-09-21, R1-1: the class the identifier arm released --------------
    # A credential-named assignment whose value is a digit-free underscore-joined
    # phrase went from MASKED to NO HIT AT ALL under every name in
    # ``IDENTIFIER_ARM_NAMES``, on every surface in ``IDENTIFIER_ARM_SPELLINGS`` —
    # nothing registered for containment either, so the later exact-value pass
    # could not contain it. The class lives in the POSITIVE half now, on the
    # alphabets and surfaces the review measured, because the coverage gap (no row
    # in this class on EITHER side) is what let it ship.
    *(
        Case(
            _arm_spelling(name, spelling, value),
            f"{spelling}: a {label} value under a credential name must be MASKED",
        )
        for name in IDENTIFIER_ARM_NAMES
        for spelling in IDENTIFIER_ARM_SPELLINGS
        for label, value in (IDENTIFIER_ARM_VALUES[0],)
    ),
    # ...and the two neighbours of that class, on the surface that carries them
    # most often. A hyphen is a separator a person writing a password reaches for
    # and the arm never touched it; a digit is the floor.
    *(
        Case(
            _arm_spelling(name, "assignment", value),
            f"the {label} neighbour of the released class, under a credential name",
        )
        for name in IDENTIFIER_ARM_NAMES
        for label, value in IDENTIFIER_ARM_VALUES[1:]
    ),
    # --- 2026-09-21, R1-3: only an ESCAPE's letters may be detached -----------
    # The first revision detached whatever followed a backslash, so a literal one
    # in front of a name ate the name's own first letter and the mask was lost
    # where origin/main had kept it. No escape spelling, so nothing may move.
    Case(
        "\\" + "PASSWORD=" + IDENTIFIER_ARM_VALUES[0][1],
        "a literal backslash before a credential name is not an escape: the mask stays",
    ),
    # ...and the spelling where a literal backslash is DECIDABLE (agent review
    # R2-F2). A rendering writes a LITERAL backslash as TWO of them, so the
    # character before the name is still a backslash and `t` is an escape letter:
    # R1-3's fix ate it anyway, `oken` is not a credential name, and the mask was
    # lost here with no hit and nothing registered — so no later exact-value pass
    # could contain the value either. Two backslashes are the one spelling where the
    # reading is not ambiguous, which is why this row is a positive rather than a
    # documented residual.
    Case(
        "\\\\" + "token=" + FIXTURE_VALUE,
        "a DOUBLED backslash is a literal one: the name after it is not an escape's",
    ),
    # --- 2026-09-21, R1-2: an escaped break must not cut the MASK -------------
    # A value whose own bytes carry an escaped break, and the value does NOT spell
    # its own name (``_repeats_its_own_name`` would spare it for the other reason).
    # The mask covers the whole run: at the revision under review the tail came
    # back READABLE while the hit still graded ``complete=True``, and when the run
    # before the break was shorter than the floor nothing was masked or registered.
    Case(
        "CLIENT" + "_SECRET=" + "alpha" + "_run" + "_body" + "\\n" + "more" + "_body" + "_material",
        "a value carrying an ESCAPED break: the whole run is masked, tail included",
    ),
    Case(
        "CLIENT" + "_SECRET=" + "run" + ">" + "\\n" + "zip" + "tail" + "material",
        "the same with a run shorter than the floor before the break: still masked",
    ),
    # --- 2026-09-22: the VALUE side of the flag-NAME judgement ----------------
    # The reported failure is a flag's argument that IS a store NAME being masked.
    # The judgement that replaced the credential-word tail check releases the NAME
    # spelling ONLY, so these are the arms it must not reach — each is a value a
    # reader hands the same flags, and each is byte-identical at ``origin/main``.
    # The two underscore-carrying rows are what a widening that dropped the CASE
    # requirement would eat, which is why they are here rather than left to a
    # differential (see ``_is_a_name_in_the_store_grammar``).
    Case(
        "--token " + "ghp_AbCd1234EfGhIjKlMnOpQr",
        "an issuer token under the token flag: lower case keeps it a VALUE",
    ),
    Case(
        "--secret " + IDENTIFIER_ARM_VALUES[0][1],
        "a digit-free underscore phrase: the R1-1 class in a flag position",
    ),
    Case(
        "--password " + IDENTIFIER_ARM_VALUES[2][1],
        "a digit-carrying underscore phrase: the same class, with a digit",
    ),
    Case(
        "--api-key " + "hunter" + "2" + "xyz",
        "a single unseparated token: no separator, so no NAME",
    ),
    Case(
        "--token=" + "dGhp" + "cyBpcyBhIHRva2Vu=",
        "a padded base64 flag value: case and a symbol keep it a VALUE",
    ),
    # --- 2026-09-22: the flag rule's WORD-shaped over-mask, at the GRADING end ---
    # The mask of a word after a flag is deliberate (the suite's own prose test says
    # why); the ESCALATION was the defect, and this row makes the corpus the referee of
    # it. The word occurs TWICE in the line, so the whole-value half of the exposure
    # question answers YES for reasons that have nothing to do with the mask — which is
    # exactly the 33 KB documentation read that demanded a rotation for the word
    # ``when``. Every positive is asserted non-escalating by
    # ``test_only_the_documented_positive_case_escalates``, so this row needs no test of
    # its own. Assembled from its segments, like the rows above, so no literal in this
    # SOURCE is a flag VALUE.
    Case(
        "--api-key " + "when you need it, and when the flag is set it wins",
        "a word after the flag, twice in one line: masked, and never escalated",
    ),
    # --- 2026-09-23: the ONE-WORD store name, which the release does not reach -----
    # ``_is_a_name_in_the_store_grammar`` [redacted] a separator, and that requirement is
    # load-bearing rather than stylistic: its first form — a run of capitals with no
    # separator — released real all-caps credential values (the rows above). The cost is
    # this one. ``normalize_credential_key`` maps an operator-typed ``prod`` to ``PROD``,
    # so a ONE-WORD store entry is a legal name the arm still masks, and the module's
    # docstring used to describe the underscore-carrying spelling as the one every
    # operator-typed key collapses to. Pinned here as a positive because the behaviour
    # that must not drift is the MASK: the release is narrower than that docstring
    # implied, the docstring is corrected in the same commit, and a later round that
    # widens the release needs this row to argue against (agent review R1-4). Assembled
    # from its parts, like the rows above, so no literal in this SOURCE is a flag VALUE.
    Case(
        _secret_run("PROD"),
        "the accepted over-mask: a ONE-WORD store name carries no separator to read",
    ),
    # The positives this change adds, and the reason they are its evidence: each is
    # a REAL credential spelled in a way that resembles a type, so the type clause
    # releasing one would be the fix eating the thing it protects.
    *TYPE_ANNOTATION_POSITIVES,
)


#: Ordinary text that must come back BYTE-IDENTICAL. Each reason names what the
#: rule that could have eaten it is, because a negative is only evidence if it
#: is the case that rule would have failed.
NEGATIVE_CASES: tuple[Case, ...] = (
    # --- placeholders: never masked, never registered (still counted) ---------
    Case("CI_JOB_TOKEN=${CI_JOB_TOKEN}", "a variable reference, not a value"),
    Case("GITLAB_TOKEN=$GITLAB_TOKEN", "the same, unbraced"),
    Case("PASSWORD=<password>", "an angle-bracket placeholder"),
    Case("PASSWORD=changeme", "the canonical placeholder word"),
    Case("PASSWORD=gitlab-ci-token", "a placeholder the platform itself uses"),
    Case("token=nonEmptyString", "a schema placeholder"),
    Case("PASSWORD=***", "a single repeated character"),
    Case("PASSWORD=%s", "a printf placeholder"),
    Case("PASSWORD={{ vault_secret }}", "a template placeholder"),
    # --- added in round 2: the false positives the widened rules produced ------
    Case('env_keys="OPENAI_API_KEY"', "a keyword argument whose value is a NAME"),
    Case('key="model_name"', "a key whose value is a NAME"),
    Case(
        'PRESERVED_USER_TURN_KEY = "compaction_preserved"',
        "a *_KEY constant holding a name",
    ),
    Case('REJECTION_CLASS_KEY = "rejection_class"', "the same shape"),
    Case('TOKENS_OBTAINED_AT_KEY = "tokens_obtained_at"', "a value that repeats its own name"),
    Case("tokens=tokens_before", "a value that repeats its own name"),
    Case("class_key = class_key", "a value that repeats its own name"),
    Case(
        'key="providers.anthropic.cache_ttl_1h_min_context_tokens"',
        "a dotted config path with digits",
    ),
    Case('_PUBLIC_LISTING_TOKEN = "public-catalogue-read"', "a lowercase hyphenated name"),
    Case("run(password=swordfish1)", "a call's keyword argument"),
    Case("return sorted(roots, key=_node_order)", "a keyword argument holding a reference"),
    Case("pypi-local-operator.json", "a filename whose tail has a dot"),
    Case('cache = tmp_path / "pypi-local-operator.json"', "the same, in a path expression"),
    Case("pypi-local-operator.json.<random>.tmp", "a dotted temp filename after the prefix"),
    Case(
        '"npm_config_update_notifier": "false"',
        "an env var NAME that starts with a vendor prefix",
    ),
    # --- 2026-09-21: the same NAME carrying its ASSIGNMENT, which the guard's
    #     lowercase-words rule had no arm for. Each of the first four below fired
    #     as a vendor token, and the first ESCALATED: a 48-character tail against
    #     a 6-character fragment window meant six-letter fragments of an ordinary
    #     name were all over the prose around it, so the mask graded EXPOSED and a
    #     session filed a rotation ticket for a package-manager toggle. The reason
    #     all of them must survive is one reason: an all-lowercase run joined by
    #     separators is a NAME, while a real issuer tail is one unbroken base64-ish
    #     run carrying mixed case and/or a digit.
    Case(
        "npm_config_manage_package_manager_versions=false",
        "an env assignment in prose: the tail is the NAME and its value",
    ),
    Case(
        "`npm_config_manage_package_manager_versions=false`",
        "the same assignment in backticks, the reported session's spelling",
    ),
    Case(
        "npm-config-manage-package-manager-versions",
        "the same NAME with the other join: a dash is the same name",
    ),
    Case(
        "npm_config_manage_package_manager_versions=11.22.0",
        "the same NAME with a version value: the dot lookahead already spares it",
    ),
    #     The four rows above cover the reported family in HALVES — the underscore
    #     join with a value, and the dash join without one. The row below is the
    #     combination, and it is the one #1399 moved: an ordinary env NAME joined by
    #     dashes CARRYING a value was read as a prefixed issuer token and masked
    #     (measured against the pre-fix module, where it fired as
    #     ``vendor-prefixed-token``), so an edit that made the ``=`` arm
    #     separator-specific would red THIS row rather than nothing at all. It is a
    #     true negative on this tree, and it carries its reason like every other row.
    Case(
        "npm-config-manage-package-manager-versions=false",
        "the same NAME with the dash join AND a value: the two rows above, combined",
    ),
    Case(
        "whsec_config_update_notifier",
        "the same NAME under a prefix that carries its own separator",
    ),
    # --- the RESIDUAL the tail rule ACCEPTS, pinned on purpose (agent review
    #     R1-1): a run of lowercase words joined by separators is a NAME whatever
    #     prefix precedes it, so a hypothetical issuer tail spelled that way is
    #     left readable. No **real** token of the shape exists in the corpus, or
    #     in the 2.6 GB of the fleet's transcripts the survey covered; the row is
    #     here so that a real one breaks a test instead of passing silently, and
    #     so the cost of the rule is visible to anyone reading the two halves
    #     together.
    Case(
        "glpat-lowercase-token-value",
        "the accepted residual: a lowercase separator-carrying tail is a NAME",
    ),
    Case("npm run build --prefix ./apps", "npm as a package manager"),
    Case("terraform output name", "the safer form the advisory itself recommends"),
    Case("cat .env.example", "an example file is not a secret"),
    Case("Basic authentication is required for the console host", "the word Basic in prose"),
    Case("# Bearer serves both API keys and OAuth access tokens on this wire", "prose"),
    Case("must carry the post-pass occupancy while retaining the pre-pass usage", "prose"),
    Case("PWD=/Users/example/project", "the shell's working directory"),
    # --- the documented over-masking traps -----------------------------------
    Case("max_tokens=262144", "a model parameter, not a credential"),
    Case("max_tokens: 8192", "the same trap in the JSON/YAML spelling"),
    Case("context_tokens=12345678", "a token COUNT, plural"),
    Case("total_tokens: 98765432", "a usage total"),
    Case("num_tokens=128000", "another count"),
    Case("prompt_tokens=1024", "a prompt count"),
    Case('"secret": "missing"', "a placeholder under a credential name (under the value floor)"),
    Case("secret: none", "a placeholder value"),
    Case("token: (unset)", "a placeholder value in parentheses"),
    Case("Authorization: Bearer", "a bearer keyword with no value at all"),
    Case("Authorization: Bearer abc", "a bearer value below the floor"),
    Case("SERVICE_URL=https://api.example.com/v1", "an ordinary endpoint under a *_URL name"),
    Case(
        "API_URL=https://api.internal/v2/items?page=3", "an ordinary endpoint with a query string"
    ),
    Case(
        "CALLBACK_REDIRECT_URI=http://localhost:8080/cb",
        "an OAuth redirect with no credential in it",
    ),
    Case("BASE_URL=http://127.0.0.1:8080", "a loopback base URL"),
    Case(
        "ENDPOINT=https://grpc.internal:443", "an endpoint with no userinfo and no credential query"
    ),
    Case("git@github.com:owner/repo.git", "a git remote, which is a user, not a credential"),
    Case("127.0.0.1:8080", "a host and port"),
    Case("postgres://", "a scheme with no userinfo"),
    Case("postgres://host", "a bare host"),
    Case("postgres://host:5432/db", "a host, port and database"),
    Case(
        "postgresql://host:5432/db?sslmode=require",
        "an ordinary DSN-shaped URL with no credentials",
    ),
    Case("redis://localhost:6379/0", "an ordinary redis URL"),
    Case("mongodb://localhost:27017/app", "an ordinary mongo URL"),
    Case("https://user@example.com/profile", "a userinfo with no password"),
    Case("user:pass@host", "a userinfo fragment with no scheme"),
    Case("The password must be rotated every 90 days.", "the word 'password' in ordinary prose"),
    Case("Ask the user for their password before proceeding.", "the same word in an instruction"),
    Case("this is a secret between us", "the word 'secret' in prose"),
    Case("The token budget is the whole point of this module.", "the word 'token' in prose"),
    Case(
        "Set client_secret to a value from the vault UI.",
        "a credential NAME in prose with no value",
    ),
    Case("max_connections=100", "a pool-size parameter"),
    Case("num_tokens_per_second=42.5", "a throughput figure"),
    Case("primary_key=id", "a database column, value under the floor"),
    Case("keyboard_layout=qwerty", "a name that merely contains 'key'"),
    Case("monkey=banana", "the same trap spelled differently"),
    Case("cache_key=user:42:profile", "a cache handle, not a credential"),
    Case("ALPHABET=abcdefghijklmnopqrstuvwxyz", "a named constant that is not a credential"),
    Case("export PATH=/usr/local/bin:$PATH", "an ordinary environment export"),
    Case("API_KEY=", "an empty assignment"),
    Case("API_KEY=short", "a value below the floor"),
    Case("docs/api_key.md", "a file path that contains a credential name"),
    Case("public_key_path=/etc/ssh/id_rsa.pub", "a public key path, nothing secret in it"),
    # --- opaque identifiers that are NOT credentials -------------------------
    Case("0123456789abcdef0123456789abcdef", "a bare hash with no issuer prefix"),
    Case("550e8400-e29b-41d4-a716-446655440000", "a UUID"),
    Case("01ARZ3NDEKTSV4RRFFQ69G5FAV", "a ULID"),
    Case("a3f5c1d9e7b24680a3f5c1d9e7b24680a3f5c1d9", "a git SHA"),
    Case(
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAA"
        "AAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
        "a base64 image header",
    ),
    Case('{"status": "ok", "count": 42, "items": ["a", "b"]}', "an ordinary JSON payload"),
    Case("https://api.example.com/v1/items?id=42&limit=20", "an ordinary API URL"),
    Case("containerPort: 8080", "a kubernetes port field"),
    Case("ports:\n  - containerPort: 8080\n    protocol: TCP", "a port block"),
    Case("redis://localhost", "a bare redis URL"),
    # --- commands that are ordinary work -------------------------------------
    Case(
        "env -i HOME=/tmp PATH=/usr/bin cmd",
        "env used to BUILD a clean environment, not to dump one",
    ),
    Case("printenv PATH", "a single harmless variable"),
    Case("printenv HOME", "the same"),
    Case("env | cut -d= -f1", "the names-only form the notice itself recommends"),
    Case("printenv | cut -d= -f1", "the same, for printenv"),
    Case("npm list --depth=0", "an ordinary npm command"),
    Case(
        "npm token list",
        "prints tokens only when tokens exist; the NOTICE covers it, the shape pass must not",
    ),
    Case("kubectl get pods -n backend-services", "an ordinary kubectl read"),
    Case("kubectl describe pod api-7d9f-xyz", "an ordinary describe"),
    Case("kubectl exec -it api -- sh -c 'ls /app'", "an ordinary exec"),
    Case("docker ps -a", "an ordinary docker read"),
    Case("docker exec -it api sh", "an ordinary docker exec"),
    Case("docker compose up -d", "an ordinary compose run"),
    Case("docker compose config --services", "the safer compose form"),
    Case("docker run -p 8080:80 nginx", "a published port, not a password"),
    Case("ssh -p 2222 host", "an ssh port, not a password"),
    Case("mysql -u root -e 'select 1'", "a mysql call with no password flag"),
    # The production misfire, measured 2026-09-19: a watch-log entry that QUOTED
    # the documented way to hand a stored secret to a child. The token after
    # ``--secret`` is the secret's NAME in the store — the thing an operator has
    # to be able to read — and it must survive byte-identical, because the hit it
    # caused filed a rotation ticket for a credential that was not in the text.
    # Joined from its segments so the literal never appears as a flag VALUE in this
    # SOURCE: a flag followed by a value is exactly the shape a redaction pass
    # rewrites, and this file is read by agents through those passes.
    Case(
        "lop secret run --secret "
        + "_".join(("OS", "PROD2", "ADMIN", "PASSWORD"))
        + " -- <command>",
        "a flag NAMING a stored secret, not carrying one",
    ),
    Case("aws sts get-caller-identity --query Account --output text", "an ordinary AWS read"),
    Case("aws configure list-profiles", "a profile listing"),
    Case("gh auth status", "a login check that prints no token"),
    Case("glab auth status", "the same with -t omitted"),
    Case("terraform plan", "an ordinary terraform invocation"),
    Case("terraform state list", "an ordinary state read"),
    Case("vault kv list secret/", "a LIST, which prints names only"),
    Case("heroku apps", "an ordinary heroku read"),
    Case("gcloud config list", "an ordinary gcloud read"),
    Case("grep -rn 'password' docs/", "a grep for the word, not a value"),
    Case("echo 'hello world'", "an ordinary echo"),
    Case("git log --oneline -5", "an ordinary git read"),
    Case("python -c 'print(1+1)'", "an ordinary python call"),
    Case("cat /etc/hosts", "reading a file that holds no credential"),
    Case("cat README.md", "the same"),
    Case("head -n 5 package.json", "the same"),
    Case("set -e", "set used as a shell option, not a dump"),
    Case("set -x", "the same"),
    Case("set -o pipefail", "the same"),
    Case("SELECT id, name FROM users WHERE active = true;", "ordinary SQL"),
    Case("curl -sS https://api.example.com/health", "an ordinary curl"),
    Case(
        "curl -H 'Accept: application/json' https://api.example.com/v1", "an ordinary curl header"
    ),
    Case("GET /v1/items?id=42 HTTP/1.1", "an ordinary request line"),
    Case("Machine learning models classify tokens.", "the word 'tokens' in prose"),
    Case("the signature parameter is required", "an API parameter name in prose"),
    Case(
        "  - name: MONGO_DSN\n    valueFrom:\n      secretKeyRef:\n       "
        " name: mongo\n        key: DSN",
        "a k8s env entry that REFERENCES a secret instead of carrying one",
    ),
    Case("The DSN is configured in the environment.", "prose about a DSN"),
    # --- the counts a widened name judgement has to keep releasing ------------
    # The measured false positive, from the Bedrock cost-tracking work: the count
    # word is the QUALIFIER and it is not always the first segment, so a judgement
    # that read ``segments[0]`` only treated ``ephemeral_5m_input_tokens`` as a
    # credential NAME, masked a counter, and — on the compact spelling, where the
    # next field's key lies inside the match — filed a rotation demand for a
    # NUMBER (the operator's write of the Bedrock evidence file, 2026-09-20).
    # Every spelling here must come back byte-identical: the pair that fired, the
    # evidence file's own line, and the ordinary companions of a usage object.
    Case(
        '{"cache_creation":{"ephemeral_5m_input_tokens":3536,' '"ephemeral_1h_input_tokens":0}}',
        "an Anthropic usage counter pair in compact JSON: a COUNT, not a name",
    ),
    Case(
        '{"ephemeral_5m_input_tokens":3536,"ephemeral_1h_input_tokens":0}',
        "the same pair at the top level of a usage object",
    ),
    Case(
        COUNTER_USAGE_LINE,
        "the evidence file's own line: a counter under a qualified name",
    ),
    Case(
        '{"cache_read_input_tokens":10,"cache_creation_input_tokens":20}',
        "the ordinary usage companions of the counter",
    ),
    # --- the tail arm's BOUNDARY, one case per name (QA round 2, Q2-1) ---------
    # The direction the tail arm RELEASES: a qualified plural-quantity tail. Names
    # of this form were masked before this change, and the reading kept here is that
    # the tail says what the name IS — the qualifier cannot turn a quantity into a
    # secret. Pinned as cases so a future widening that masks them again has to
    # argue with the table; the name is in the reason so each has its own test id.
    *(
        Case(
            f"{name}={FIXTURE_VALUE}",
            f"a qualified quantity tail the tail arm releases: {name} is a count",
        )
        for name in COUNT_TAIL_RELEASED_NAMES
    ),
    # --- 2026-09-21: the MASK MARKER itself, which a peer session reported three
    #     times in one hour. The third report was this claim: a message CONTAINING
    #     the marker re-fires a fresh incident — "a detector whose output is its own
    #     input cannot settle". Measured false on this table's own tree, and unpinned
    #     by the table until now, which is why the claim travelled as a message
    #     instead of meeting a row. Each of the three shapes below is byte-identical
    #     on all seven surfaces, files no hit, takes no label and does not escalate.
    #
    #     THAT IS A PROPERTY OF THESE THREE TEXTS, NOT OF THE MARKER, and the
    #     difference is the whole of agent review R1-1: the section first stated the
    #     wider generalisation — that a text carrying the marker "holds no credential
    #     material for a rule to grade" — and it is measurably FALSE. What decides it
    #     is a value CLASS wide enough to hold the marker's own brackets. The rule
    #     matches, files a hit and labels the text, and then rewrites the marker back
    #     to ITSELF, so the bytes survive and neither half here can hold such a
    #     spelling: a POSITIVE row asserts its text was rewritten, and a NEGATIVE row
    #     asserts the pass left it alone in every OBSERVABLE way — the hit list is the
    #     observable that moves.
    #
    #     The corrected boundary, measured 2026-09-21 by taking every POSITIVE row's
    #     masked form (``scrub_shapes_with_hits(text)[0]``) and driving THAT back
    #     through ``match_shape_names``: 23 rows re-fire, over EIGHT labels, and every
    #     one of the 23 is byte-stable on all seven surfaces with ``reached_model``
    #     False. Agent review R2-1 counted the first version of this list and it was
    #     short on BOTH halves — seven labels over ten spellings against its own
    #     nine-item enumeration — so what follows is the measurement rather than a
    #     recollection. Each family is spelled with the marker in its VALUE position,
    #     which is what this class IS: the rule matches, files a hit, labels the text
    #     and then rewrites the marker back to itself. By label, with the row count
    #     each accounts for:
    #     ``npmrc-auth-token`` (3): ``_authToken=[redacted]``,
    #     ``//registry.npmjs.org/:_authToken=[redacted]`` and the base64 ``_auth=[redacted]``
    #     ``netrc-password`` (2): ``machine api.github.com login robot password [redacted]``
    #     ``machine example.com login bob password [redacted]``
    #     ``cookie-header`` (3): ``Cookie: [redacted]``, ``Set-Cookie: [redacted]``
    #     and a cookie header inside a curl invocation
    #     ``client-inline-password`` (3): ``redis-cli -a [redacted]``, ``mongosh -p[redacted]``
    #     ``mysql -u root -p[redacted]``
    #     ``docker-login-password`` (1): ``docker login -u robot -p [redacted]``
    #     ``openssl-pass-phrase`` (3): the bare ``-pass [redacted]``, ``-passin [redacted]``
    #     and the ``pass: [redacted]`` form
    #     ``curl-user-credential`` (1): ``curl -u user:[redacted]``
    #     ``credential-query-param`` (7): ``api_key=[redacted]``, ``token=[redacted]``,
    #     ``access_token=[redacted]``, ``secret=[redacted]``, ``password=[redacted]``,
    #     ``signature=[redacted]`` and a JSON ``apikey`` spelling
    #     That is eight labels over 23 spellings, every row carrying exactly one.
    #     The ROWS are a closed set, because they are the corpus rows that re-fire
    #     today; the LABELS are not, because a rule added tomorrow claims a row
    #     without anyone editing this comment. The class is already driven through
    #     the shipped path by ``test_a_marker_valued_hit_no_longer_escalates``.
    #
    #     DEFERRED, and PRE-EXISTING rather than introduced here: the marker spelled
    #     as the VALUE of a lower-case assignment does NOT survive.
    #     ``password="[redacted]"`` comes back as ``[redacted]"`` — the NAME and its
    #     separator are DELETED, with 0 hits, 0 labels and no notice, on all seven
    #     surfaces (``api_key="[redacted]"`` likewise). ``_INCOMPLETE_MASK_LEFT_RE``
    #     reads the ``name=`` run as the readable HEAD of a credential whose mask
    #     stopped at a quote — its NAME exclusion is capitals-only, and its run class
    #     carries ``=`` — and ``_close_partial_masks`` records no hit, so the notice
    #     path never learns the text was rewritten. A rewrite with no label is exactly
    #     the "blinding the agent to the text it is reading" case this half exists to
    #     prevent, which is why it is recorded rather than left implicit; it is NOT
    #     fixed here because the production diff of this commit against the PR's merge
    #     base ``2a9a737a`` is empty (tests only) and the fix is a production change, so
    #     the rows below
    #     are narrowed to what they measure rather than widened to answer for it.
    Case(
        "[redacted]",
        "the mask marker alone: the pass's own output is not its own input",
    ),
    Case(
        "The alert quoted MONGO_DSN=mongodb+srv://svc:[redacted]@db.invalid/x as " "its evidence.",
        "the marker inside prose, in a credential position: a peer's evidence",
    ),
    Case(
        '{"command": "grep -rn \'MONGO_DSN=[redacted]\' notes/"}',
        "the marker inside a JSON tool argument: the shape a bash call journals",
    ),
    # --- 2026-09-21, second wave: the ESCAPED rendering is a surface too ------
    # An assignment is scrubbed on more than one surface, and a tool call's
    # journaled spelling is its JSON payload: every newline inside the string
    # arrives as the two characters ``\`` and ``n``. Two defects came out of
    # that one fact, both measured on a ``write`` of ordinary Python source whose
    # file on disk has no shape in it at all (the incident that named them).
    #
    # (1) The escape's OWN LETTER was absorbed by the name group. The count-trap
    #     exclusion keys on the FIRST segment of a name, so a name beginning
    #     immediately after an escaped newline arrives with the newline's letter
    #     glued to its front — and stops being a count. The two blank lines before
    #     the next ``def`` supplied the second escape.
    # (2) The value ran ACROSS those escapes: ``[^\s]`` "already cannot cross a
    #     line" everywhere except in a rendering, where the line is two
    #     non-whitespace characters, so a constant became a 13-character value.
    # Built from its parts, like the other rows this file cannot spell: the joined
    # spelling is exactly what the pass rewrites, and this file is read through it.
    Case(
        PRE_ESCAPED_LINE
        + "MAX"
        + "_TOKENS = 4096"
        + "\\n" * 3
        + "def load_vendor_keys() -> dict[str, str]:",
        "the count trap one escaped newline away, as a JSON payload spells it",
    ),
    Case(
        PRE_ESCAPED_LINE + "context_tokens=12345678" + "\\n" * 3 + "def run() -> None:",
        "the same trap with a value long enough to clear the value floor",
    ),
    Case(
        PRE_ESCAPED_LINE + "API" + "_KEY = PLACEHOLDER" + "\\n" * 3 + "def run() -> None:",
        "the same run under a name the count trap does not cover",
    ),
    # --- an IDENTIFIER value is a NAME, whatever name it sits under ------------
    # The identifier clause required a LEADING underscore, so a value that is
    # plainly the name of another local — spelled the way this tree spells locals
    # — was masked, and when the same identifier appeared again anywhere in the
    # file the mask graded EXPOSED and escalated. Both rows are verbatim from
    # ``local_operator/providers/clients.py``, whose ``read`` filed a rotation
    # notice for them on 2026-09-21.
    Case(
        "reasoning_tokens=" + "thought_tokens",
        "a usage counter assigned another local: an identifier, not a secret",
    ),
    Case(
        "extra_native_tokens=" + "echo_turns" + " * reasoning_echo_placeholder_tokens()",
        "the same, in the expression that reported it",
    ),
    # --- a flag that NAMES a stored secret, in the guide's own spelling --------
    # ``--secret NAME=VAR`` is the documented way to rename a stored secret for a
    # child, and the clause that exempts a NAME tested the value as a SINGLE
    # token, so the two-part form fell straight through it and masked. The
    # harness's own ``guide://credentials`` carries this line verbatim; reading the
    # guide masked it and filed an escalated incident naming no shape at all.
    Case(
        "lop secret run --secret "
        + "GITHUB"
        + "_"
        + "TOKEN"
        + " --secret "
        + "NPM"
        + "_"
        + "TOKEN"
        + "="
        + "NODE"
        + "_"
        + "AUTH"
        + "_"
        + "TOKEN"
        + " -- npm publish",
        "a flag NAMING a stored secret under a second NAME, as the guide spells it",
    ),
    # --- a path is a NAME, and a slash is one of its separators -----------------
    # The tail after an issuer-looking prefix is a lowercase org/repo path: a NAME
    # by the same predicate that already spares the ``_`` and ``-`` spellings of
    # the identical string. ``/`` was simply missing from the separator class, so
    # only the spelling a path is ACTUALLY written with masked. Prose from a
    # docstring in ``local_operator/providers/clients.py`` (line 2298), which is
    # what a ``read`` of that file reported as ``vendor-prefixed-token``.
    Case(
        "xai" + "-org/" + "grok" + "-build",
        "a vendor-looking prefix in front of an org/repo PATH: a slash-joined NAME",
    ),
    Case(
        "and " + "xai" + "-org/" + "grok" + "-build" + " sends it on every request.",
        "the same slug in the docstring prose that reported it",
    ),
    # --- 2026-09-21, R1-1: the boundary the narrowed arm draws ----------------
    # The arm is scoped to a QUANTITY-NOUN tail, which is the judgement the two
    # reported counters share — a plural quantity noun IS the quantity, so a usage
    # counter's value is the name of another local. The boundary it leaves is real
    # and it is pinned HERE rather than left to a paragraph: a passphrase spelled
    # as an identifier, under a name whose tail is a quantity noun, is read as a
    # NAME. Every other credential name keeps masking it (see the positive block).
    # ``is_count_shaped`` does not cover these names — its tail arm wants a count
    # word in the middle as well — which is why the arm is what spares them.
    *(
        Case(
            name + "=" + IDENTIFIER_ARM_VALUES[0][1],
            "the narrowed arm's boundary: a quantity-noun tail is a count, not a secret",
        )
        for name in ("API" + "_TOKENS", "NPM" + "_TOKENS", "REASONING" + "_TOKENS")
    ),
    # --- 2026-09-21, R1-4: the escapes a JSON rendering really writes ---------
    # The count trap one escape away, in the spelling ``json.dumps`` writes for the
    # characters it cannot spell in two: a raw U+2028 is a line break to
    # ``str.splitlines`` and arrives as its own six-character spelling, so the
    # false positive reproduced there too. Percent-encoding is deliberately NOT
    # handled — no renderer on these surfaces writes one, and the pattern says so.
    # ...and one the pre-escape judgement RELEASES, pinned rather than left to a
    # reviewer to find: a value whose first line is a digit-free camelCase NAME is
    # spared by the type-name clause, and with the escape read as the line break it
    # is, the same assignment is spared on a REAL newline by ``origin/main`` too.
    # So the escaped surface now agrees with the real one instead of masking more
    # than it; the tail that used to come with it is no longer swallowed. The mask
    # still covers the whole run whenever the first line IS a credential, which is
    # the case the R1-2 fix is about.
    Case(
        "CLIENT" + "_SECRET=" + "alpha" + "Run" + "Body" + "\\n" + "more" + "Body" + "Material",
        "a type-name-looking first line: the escape is a line break for the judgement too",
    ),
    *(
        Case(
            "RESULTS = os.path.join(HERE)"
            + escape
            + "MAX"
            + "_TOKENS = "
            + str(4096)
            + escape
            + "def run() -> None:",
            f"the count trap under the six-character escape {escape}",
        )
        for escape in ("\\u2028", "\\u000a")
    ),
    # --- 2026-09-22: a secret NAME in a credential-flag position --------------
    # The reported failure, measured on the installed build: the command the
    # credentials guide teaches, run with a store entry named after the SYSTEM it
    # belongs to (the tail is USERNAME, which is not one of the credential words
    # the guard used to require). Every tool result masked that name, so the script
    # an agent then authored from the displayed text asked the store for a secret
    # literally named ``[redacted]``. It cost no incident — a mask is not an
    # escalation — which is why only the workflow caught it. Joined from its
    # segments, like the 2026-09-19 row above, so no literal in this SOURCE is a
    # flag VALUE.
    Case(
        _secret_run("MINERVA_UI_NPROD_USERNAME --secret MINERVA_UI_NPROD_USERNAME"),
        "the guide's own publish command: a store NAME under the flag, twice",
    ),
    Case(
        "--secret " + "MINERVA_UI_NPROD_USERNAME",
        "a bare store NAME after the secret flag: the spelled-alone case",
    ),
    Case(
        "--token " + "MINERVA_UI_NPROD_USERNAME",
        "a store NAME under the token flag: the argument position decides",
    ),
    Case(
        "--password " + "MINERVA_UI_NPROD_USERNAME",
        "a store NAME under the password flag: the same",
    ),
    Case(
        "--api-key " + "MINERVA_UI_NPROD_USERNAME",
        "a store NAME under the api-key flag: the same",
    ),
    Case(
        "--secret=" + "MINERVA_UI_NPROD_USERNAME",
        "the = spelling of the same argument: one verdict, whichever binds it",
    ),
    Case(
        _secret_run("MINERVA_UI_NPROD_USERNAME=MINERVA_UI_NPROD_STORE"),
        "the two-part form whose right half is not a credential word either",
    ),
    Case(
        "cat publish.sh" + "\n" + _secret_run("OS_PROD2_ADMIN_HOST"),
        "a cat of the script the agent authored from the displayed text",
    ),
    Case(
        "grep -n secret publish.sh" + "\n" + "4:" + _secret_run("OS_PROD2_ADMIN_HOST"),
        "a grep rendering of the same line",
    ),
    # The residual, pinned rather than left to a differential: this spelling IS a
    # store entry's name, and an arm that reads the argument by shape cannot tell it
    # from a credential someone chose in caps. Released deliberately; the value side
    # above is what keeps the release narrow.
    Case(
        "--token " + "ABC_123_XYZ",
        "the accepted residual: an all-caps underscore token is read as a NAME",
    ),
    # --- 2026-09-23: the TWO-PART spelling of the same residual, which had no row ---
    # ``--secret [redacted] is the flag's documented grammar, so BOTH halves are
    # references and each is read by its env-name shape alone. The arm is therefore
    # WIDER than the one-part release above, which does require an underscore, and the
    # width is the grammar's rather than an accident: the left half is a store entry's
    # name, and the store leaves a ONE-WORD key with no separator (``prod`` is an entry,
    # ``normalize_credential_key`` and all), while the right half is the child's variable,
    # whose conventional spelling has none either (``PGPASSWORD``). Two rows, because the
    # two spellings behave DIFFERENTLY against the module this change replaced, and both
    # readings are measured rather than argued:
    #
    #   * caps halves WITH separators are clean on both modules — pinned so the two-part
    #     reading is a rule someone wrote down rather than a side effect of the halves
    #     happening to look like names;
    #   * caps halves with NO separator in either are MASKED at ``5e799c4c`` and released
    #     here. That is the residual agent review R1-3 measured, and it had no row
    #     anywhere: a differential found it rather than the corpus stating it. It is the
    #     sharper of the two releases, which is why it is the row the finding asked for.
    #
    # Both released deliberately; the value side above (issuer tokens, padded base64,
    # digit-carrying phrases) is what keeps the release narrow.
    Case(
        "--token " + "PROD" + "=" + "PGPASSWORD",
        "a two-part argument with a separator in each half: read as references, as at base",
    ),
    Case(
        "--token " + "ABCDEF" + "=" + "ABCDEF",
        "no separator in EITHER half: masked at the base module, released by this change",
    ),
    # The spellings this change releases, as rows rather than as a paragraph.
    *TYPE_ANNOTATION_NEGATIVES,
    # ...and the one it releases that is NOT a type annotation, pinned here so the
    # boundary the type clause draws is a row a later change has to argue with.
    #
    # ``Pass<Word>`` is a credential spelled EXACTLY as ``Ident<Ident>`` — capitals
    # on both sides, no digit, no symbol, no word break — and no spelling test can
    # tell that from a type application: a type name and a chosen password use the
    # same alphabet. It sits in the NEGATIVE half deliberately, which is this file's
    # convention for an accepted residual (see the lowercase vendor tail above), and
    # the residual is bounded by construction rather than by hope: no issuer's
    # alphabet contains ``<`` or ``>`` — base64url, hex, JWT and UUID all exclude
    # them — every vendor prefix is lowercase, and the escape routes a credential
    # actually has (a rendered blob, a DSN, a header) all carry the symbols the
    # whole-value confinement check rejects.
    #
    # The residual is now the WHOLE-CONDITION spelling, and the boundary is on both
    # sides: a digit in either the base or the argument masks, and so does a word
    # break in either. ``Correcthorse<Battery>`` is the same value with its
    # underscore dropped — digit-free, no word break — and it is pinned here for the
    # same reason: the arm releases it, so it is a boundary rather than an accident.
    Case(
        "API" + "_KEY=" + _generic("Pass", "Word"),
        "the accepted residual: Ident<Ident> is a type by spelling, and is recorded",
    ),
    Case(
        "DB" + "_PASSWORD=" + _generic("Correcthorse", "Battery"),
        "the residual's no-word-break spelling: underscore removed, still digit-free",
    ),
)


@dataclass(frozen=True)
class DumpCase:
    """One credential-printing command shape: what fires, and why."""

    command: str
    fires: bool
    reason: str


#: One case per CLI in the detector table, plus the ordinary commands the
#: detector must leave alone. ``fires=False`` entries are the ones that make the
#: table trustworthy: a notice that nags about ``npm list`` is a notice the
#: agent learns to ignore.
DUMP_COMMAND_CASES: tuple[DumpCase, ...] = (
    DumpCase("env", True, "bare env prints every value in the environment"),
    DumpCase("printenv", True, "bare printenv prints every value"),
    DumpCase("set", True, "bare set prints every shell variable"),
    DumpCase("env | grep -iE '^(PORT|MONGO)'", True, "the incident's own pipeline"),
    DumpCase("cd /app && env", True, "env after a chained command"),
    DumpCase("printenv MONGO_DSN", True, "a single variable whose name says it is a credential"),
    DumpCase("printenv GITHUB_TOKEN", True, "a single credential variable"),
    DumpCase("kubectl exec -n backend-services api-0 -- env", True, "kubectl exec env"),
    DumpCase("kubectl exec -it api -- sh -c 'env'", True, "kubectl exec with a shell"),
    DumpCase("kubectl get secret mongo -o yaml", True, "a secret read in yaml"),
    DumpCase(
        "kubectl get secrets -n backend-services -o json", True, "every secret in a namespace"
    ),
    DumpCase("kubectl describe secret mongo", True, "a secret describe"),
    DumpCase("docker inspect api-0", True, "docker inspect prints the container's Env"),
    DumpCase("docker exec api-0 env", True, "docker exec env"),
    DumpCase("docker compose config", True, "compose config resolves and prints the environment"),
    DumpCase("aws configure list", True, "prints the configured credential material"),
    DumpCase("aws configure get aws_secret_access_key", True, "reads one credential"),
    DumpCase(
        "aws secretsmanager get-secret-value --secret-id prod/db", True, "prints a secret value"
    ),
    DumpCase("aws sts get-session-token", True, "mints and prints a temporary credential"),
    DumpCase("gcloud auth print-access-token", True, "prints an access token"),
    DumpCase("gcloud auth print-identity-token", True, "prints a token"),
    DumpCase("gcloud secrets versions access latest --secret=db", True, "prints a secret version"),
    DumpCase("gh auth token", True, "prints the GitHub token"),
    DumpCase("glab auth status -t", True, "prints the GitLab token"),
    DumpCase("vault kv get secret/prod/db", True, "prints the stored values"),
    DumpCase("vault kv read secret/prod/db", True, "the read spelling"),
    DumpCase("heroku config", True, "prints every config var"),
    DumpCase("heroku config:get DATABASE_URL -a app", True, "prints one config var"),
    DumpCase("terraform output", True, "prints every output, sensitive ones on old state"),
    DumpCase(
        "terraform state show module.db.aws_db_instance.main", True, "prints resource attributes"
    ),
    DumpCase("npm token list", True, "prints the tokens themselves"),
    DumpCase("cat ~/.netrc", True, "reads a credential file"),
    DumpCase("cat ~/.npmrc", True, "the same"),
    DumpCase("cat ~/.docker/config.json", True, "the same"),
    DumpCase("cat ~/.kube/config", True, "the same"),
    DumpCase("cat mcp.json", True, "the MCP config carries resolved credentials"),
    DumpCase("cat credentials.env", True, "the same"),
    DumpCase("cat /etc/ssl/private/server.pem", True, "reads a private key"),
    DumpCase("cat gcp-service-account.json", True, "reads a service-account key"),
    DumpCase("head -c 200 secrets/service-account-prod.json", True, "reads a key through head"),
    DumpCase("env | cut -d= -f1", False, "the names-only form the notice recommends must not fire"),
    DumpCase("printenv | cut -d= -f1", False, "the same for printenv"),
    DumpCase("env -i HOME=/tmp PATH=/usr/bin cmd", False, "env used to build a clean environment"),
    DumpCase("printenv PATH", False, "a single harmless variable"),
    DumpCase("printenv HOME", False, "the same"),
    DumpCase("npm list --depth=0", False, "ordinary npm work"),
    DumpCase("npm view lodash version", False, "ordinary npm work"),
    DumpCase("kubectl get pods -n backend-services", False, "ordinary kubectl read"),
    DumpCase("kubectl logs api-0", False, "ordinary kubectl read"),
    DumpCase(
        "kubectl get secret mongo -o jsonpath='{.data.key}'", False, "the safer single-key form"
    ),
    DumpCase("docker ps -a", False, "ordinary docker read"),
    DumpCase("docker exec -it api sh", False, "ordinary docker exec"),
    DumpCase("docker compose config --services", False, "the safer compose form"),
    DumpCase(
        "aws sts get-caller-identity --query Account --output text", False, "the safer scoped read"
    ),
    DumpCase(
        "aws sts get-caller-identity",
        False,
        "prints an account id and an ARN — identity, not a credential, so the "
        "notice would be noise; the token-ISSUING sts reads still fire",
    ),
    DumpCase("aws configure list-profiles", False, "a profile listing prints no credentials"),
    DumpCase("gcloud config list", False, "ordinary gcloud read"),
    DumpCase("gh auth status", False, "prints the login, not the token"),
    DumpCase("glab auth status", False, "the same without -t"),
    DumpCase("vault kv list secret/", False, "lists names only"),
    DumpCase("vault status", False, "ordinary vault read"),
    DumpCase("heroku apps", False, "ordinary heroku read"),
    DumpCase("terraform plan", False, "ordinary terraform work"),
    DumpCase("terraform state list", False, "lists names only"),
    DumpCase("cat /etc/hosts", False, "a file with no credential"),
    DumpCase("cat README.md", False, "the same"),
    DumpCase("head -n 5 package.json", False, "the same"),
    DumpCase("grep -rn 'password' docs/", False, "a grep for the word"),
    DumpCase("git status", False, "ordinary git work"),
    DumpCase("ls -la ~/.ssh", False, "a directory listing, not a file read"),
    # --- the file rule's SEPARATOR cases --------------------------------------
    #
    # Harvested from 39,111 real bash commands in this machine's session store: 47
    # of the 74 the file rule fired on had the credential filename in a DIFFERENT
    # simple command, in a comment, or in a heredoc body than the reading verb —
    # ordinary work nagged for a token it merely MENTIONED (the operator's words:
    # "remove the redaction warning just from credentials being used in bash").
    # The five below are those firings, masked where a path was private. The second
    # is the case agent review R1 finding 3 named: as harvested it carried no `.env`
    # at all, so it passed whatever the rule did. The fold kept main's spelling of
    # that row — the token arrives after the command substitution CLOSES, in a later,
    # unrelated command — over the `.root` / `cp .env /tmp/x` spelling our own
    # remediation wrote, so the row is named here by its text, not by an ordinal;
    # what it pins is that a `.env` past the `)` is not read.
    DumpCase(
        "head -30; echo ---; ls -la .env",
        False,
        "the read is of something else; `.env` is a later, unrelated command",
    ),
    DumpCase(
        'cat /tmp/qa/app.log); cd "$QA/tree" && grep -n "VITE_" src/app.ts; ls .env',
        False,
        "the command substitution closes before the `.env`, which belongs to a "
        "later, unrelated command",
    ),
    DumpCase(
        "tail -f app.log # writes .env",
        False,
        "a comment is not a read",
    ),
    DumpCase(
        "head -5 README.md; ls -la a.pem",
        False,
        "the token belongs to another command on the line",
    ),
    DumpCase(
        "ls -la .env* 2>/dev/null | head; echo ---; git check-ignore .env",
        False,
        "the read is the listing; `.env` is an argument to check-ignore",
    ),
    # ...and the genuine reads the same harvest found, which must keep firing: a
    # chained command, a path reached through a variable, and flags then the file.
    DumpCase("cd /app && cat .env", True, "a chained read of an env file is a read"),
    DumpCase("tail -c 120 .env | tr -d '\\n'", True, "flags, then the file"),
    DumpCase("head -c 200 ~/tmp/openrouter_key.env", True, "a key file read through head"),
    # --- a paren is not a command position on its own (2026-09-21) -------------
    # Measured on a peer session's own count-only scan, and it is the circular
    # case this table is here to make explicit: the guard matched the shape of
    # the SEARCH QUERY the agent had just written rather than anything the
    # command read. ``set`` is a shell builtin that dumps variables AND a Python
    # builtin; only the shell reading is a dump, and the paren before it was the
    # only thing saying "shell".
    DumpCase(
        "where=collections.defaultdict(set)",
        False,
        "a Python call: the paren is not a shell command position",
    ),
    DumpCase("x = set()", False, "the same builtin with no paren before it"),
    DumpCase(
        "hits=collections.Counter(); where=collections.defaultdict(set)",
        False,
        "the peer session's own scan, verbatim",
    ),
    DumpCase("( env )", True, "a real subshell: the spaced spelling still fires"),
    DumpCase("x=$(set)", True, "a command substitution still dumps"),
)
