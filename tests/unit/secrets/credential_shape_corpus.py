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


#: A credential spelled the way something spells one. Every one of these must
#: come back with at least one shape masked.
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
    # The five below are those firings, masked where a path was private.
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
)
