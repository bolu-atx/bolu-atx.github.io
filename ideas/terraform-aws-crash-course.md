# Terraform + AWS Crash Course

This is the version I wish somebody had handed me before I had to reason about a real company setup.

It is not a step-by-step tutorial. It is the mental model.

The goal is to make the pieces feel less arbitrary:

- why Terraform exists
- what AWS services are trying to optimize for
- why IAM roles matter so much
- how a normal web app maps onto AWS
- how a runner / worker system maps onto AWS
- how deployment works when GitHub Actions, Terraform, containers, and AWS are all involved

If you remember one thing, remember this:

> Terraform is how you declare what infrastructure should exist.  
> AWS is the pile of primitives that infrastructure is built from.  
> IAM is how those primitives are allowed to talk to each other without hardcoded credentials.

## 1. The Core Mental Model

When people first learn Terraform, they often think:

"Oh, this is a scripting tool for AWS."

That is the wrong model.

Terraform is closer to a planner than a script runner.

You do not tell it:

1. create VPC
2. create subnets
3. create database
4. create server

You tell it:

"I want a VPC, subnets, database, and server with these properties."

Terraform then:

- builds a dependency graph
- compares your code to its current state file
- compares that to the real cloud provider state
- computes a plan
- applies the delta

That design intent matters a lot.

Terraform is not trying to be:

- a deployment shell script
- a runtime control plane
- an application orchestrator
- a schema migration tool
- a general-purpose programming language

Terraform is trying to be:

- declarative
- repeatable
- reviewable
- environment-safe
- provider-agnostic enough to express resources consistently

That is why Terraform code can feel "stiff" compared to writing Python or Bash. That stiffness is the point.

It wants infrastructure changes to be:

- explicit
- diffable
- peer reviewed
- not hidden behind somebody clicking around in the console at 11:30 PM

## 2. Why Companies Prefer Terraform Over Console / CLI

The console is fine for learning.

It is bad as a source of truth.

The AWS console optimizes for:

- discovering services
- trying things quickly
- operating existing resources
- debugging a live system

It does not optimize for:

- consistency across engineers
- reproducibility
- knowing why a resource exists
- safe multi-environment rollout

The AWS CLI is better, but it still has the wrong center of gravity.

The CLI optimizes for:

- imperative operations
- ad hoc automation
- one-off fixes
- scripting

Terraform exists because companies eventually want this:

- infrastructure described in Git
- pull requests for infra changes
- environments that match each other structurally
- fewer "we manually set that once and forgot"
- fewer invisible settings in the console

The console and CLI still matter.

You still use them to:

- inspect live state
- debug issues
- verify assumptions
- perform emergency operations

But the ideal is:

- **Terraform creates and owns the thing**
- **the console helps you inspect the thing**

That distinction is the heart of most infra teams.

## 3. Desired State vs Runtime Control

This is one of the biggest conceptual separations in modern infra.

Terraform answers:

- What resources should exist?
- How should they be connected?
- What permissions should exist?
- What networking should exist?
- What buckets, databases, queues, roles, clusters, and distributions should exist?

Terraform does **not** answer:

- Which HTTP request should go where right now?
- Which job should run next?
- Which container should scale up because CPU is high?
- Which query should be retried?
- Which migration should run on startup?

Those are runtime concerns.

So a useful split is:

- Terraform provisions the platform
- your app and AWS managed services run on top of it

Examples:

- Terraform creates an ECS service
- ECS keeps tasks alive at runtime

- Terraform creates an RDS database
- PostgreSQL handles connections and transactions at runtime

- Terraform creates a Batch queue
- your runner submits jobs to it at runtime

If you confuse these layers, AWS feels chaotic. Once you separate them, it gets simpler.

## 4. The Three Main Things Terraform Tracks

Terraform is always juggling three realities:

### 4.1. Your config

The `.tf` files.

This is your declared intent.

### 4.2. State

The Terraform state file is Terraform's memory of what it believes it created.

This is the part beginners underestimate.

Without state, Terraform would not know:

- whether a resource already exists
- which real-world object corresponds to a resource block
- whether a rename means "replace" or "modify"

State is why Terraform can do a plan instead of just re-running everything from scratch.

### 4.3. Real provider state

What AWS actually has right now.

Terraform talks to AWS and asks:

- does this VPC exist?
- what is this bucket configured as?
- what is the ARN of this role?

Then it compares all three worlds and computes a diff.

That is the core loop.

## 5. Why State Is Such a Big Deal

A lot of Terraform weirdness becomes obvious once you respect state.

State is:

- necessary
- fragile if treated casually
- often shared across engineers and CI
- extremely important to protect

Design intent:

- Terraform needs a durable record of what it manages
- teams need a shared source of infra truth

That is why companies store state remotely instead of on laptops.

Typical pattern:

- state in S3
- locking enabled
- CI and engineers both use the same backend

Once you accept that Terraform is stateful, a lot of its choices make more sense.

## 6. Why Modules Exist

A Terraform module is just a reusable chunk of Terraform.

People often overcomplicate this.

A module is not magical.

It is just:

- a folder
- with `.tf` files
- that takes inputs
- and produces outputs

Why modules exist:

- avoid copy/paste
- create a stable internal interface
- encode your company's preferred patterns
- separate concerns

A clean mental model is:

- root module = the environment composition layer
- child modules = reusable building blocks

Example:

- `modules/network`
- `modules/database`
- `modules/frontend`
- `modules/api_service`
- `modules/runner_service`
- `modules/job_batch`

Then your environment says:

"Use one network module, one database module, one frontend module, one API module, one runner module."

That is much easier to reason about than one 2,000-line `main.tf`.

## 7. What AWS Is Really Offering You

AWS looks like a zoo until you group services by role.

For most app architectures, you can bucket things into:

### 7.1. Identity and permission

- IAM roles
- IAM policies
- trust relationships
- KMS

### 7.2. Networking

- VPC
- subnets
- route tables
- internet gateways
- NAT gateways
- security groups
- load balancers

### 7.3. Compute

- Lambda
- ECS / Fargate
- EC2
- Batch

### 7.4. Data

- RDS / Aurora
- DynamoDB
- S3
- ElastiCache / Redis

### 7.5. Edge / delivery

- CloudFront
- Route 53
- ACM
- API Gateway

### 7.6. Orchestration / events

- SQS
- SNS
- EventBridge
- Step Functions
- Batch queues

Once you group services this way, infra discussions become clearer.

Instead of "which AWS thing do I click?", you start asking:

- What compute model fits this workload?
- What identity model should this component use?
- Is this persistent data or transient compute?
- Does this thing need public reachability?
- Is this request/response traffic or background work?

That is the right level of thinking.

## 8. The Most Important AWS Concept: Identity Beats Secrets

This is the biggest jump from local dev to real cloud systems.

Locally, a lot of people are used to:

- environment variables
- `.env` files
- long-lived API keys
- one shared DB password
- one shared AWS key pair

That is workable for a laptop.

It is weak design in production.

In AWS, the preferred pattern is:

- assign an identity to the running workload
- attach permissions to that identity
- let AWS issue short-lived credentials automatically

That is what IAM roles are for.

This leads to a very important distinction:

### 8.1. Secrets for your app

Examples:

- third-party API token
- Stripe key
- OAuth client secret
- SMTP password

These are still secrets.

### 8.2. Credentials for AWS itself

Examples:

- S3 access
- reading from SQS
- writing CloudWatch logs
- launching Batch jobs
- reading from Secrets Manager

These should usually be done with IAM roles, not hardcoded AWS keys.

That is why in AWS land people keep saying things like:

- "use the Lambda execution role"
- "use the ECS task role"
- "use an instance profile"
- "use GitHub OIDC"

The design intent is:

- **workloads should identify themselves**
- **not carry permanent cloud passwords around**

That is a much stronger model.

## 9. IAM Roles: The Deep Intuition

An IAM role is easiest to understand as:

"a named bundle of permissions that some trusted principal is allowed to assume."

There are two sides to every role:

### 9.1. Trust policy

Who is allowed to assume the role?

Examples:

- Lambda service
- ECS tasks
- EC2 instances
- GitHub Actions via OIDC

### 9.2. Permission policy

What can the role do after it is assumed?

Examples:

- read from S3
- connect to RDS
- submit a Batch job
- pass a task role
- read a secret

That split is extremely important.

A role is not just "permissions." It is:

- who may become this identity
- what this identity may do

Once you see that, IAM stops feeling like pure bureaucracy.

## 10. How a Normal Web App Maps to AWS

Take a standard app:

- React frontend
- FastAPI backend
- Postgres
- file uploads

Locally this might be:

- `frontend` container
- `api` container
- `postgres` container
- maybe `nginx`

In AWS, that usually becomes:

- React static build in S3
- CloudFront in front of S3
- FastAPI running in Lambda or ECS
- RDS for PostgreSQL
- S3 bucket for uploads

Why?

Because AWS wants you to separate concerns:

- static assets do not need a long-lived server
- databases should be managed separately from app containers
- CDN delivery should happen at the edge
- app compute should not also be your file store

The mapping looks like this:

```text
Browser
  -> CloudFront
    -> S3 (frontend assets)

Browser / frontend
  -> API endpoint
    -> FastAPI compute
      -> RDS PostgreSQL
      -> S3 uploads bucket
```

That is the first useful mental picture.

## 11. Choosing Compute: Lambda vs ECS vs EC2

This is one of the first real architecture choices.

### 11.1. Lambda

Lambda is good when you want:

- request/response functions
- low ops overhead
- scale-to-zero
- event-driven compute
- short-ish workloads

Lambda is less pleasant when you want:

- long-running processes
- continuous polling loops
- lots of open database connections
- heavy container/runtime assumptions
- complicated networking edge cases

The deeper point:

Lambda is optimized for **ephemeral compute units**, not "a server but I don't want to call it a server."

### 11.2. ECS / Fargate

ECS is good when you want:

- containerized services
- long-running processes
- more normal app-server behavior
- explicit CPU/memory sizing
- easier fit for worker services and APIs

Fargate is the serverless flavor of ECS:

- you still define tasks and services
- AWS manages the underlying hosts

ECS/Fargate is often the clean fit for:

- FastAPI APIs
- background workers
- runners
- scheduled tasks

### 11.3. EC2

EC2 gives the most control:

- full VM
- full OS access
- best when you need host-level behavior

It is also the most operationally heavy.

A good heuristic:

- Lambda for small event-driven units
- ECS/Fargate for containerized services and workers
- EC2 when you truly need machine-level control

## 12. Why Static Frontends Usually Go to S3 + CloudFront

A React frontend build is just files:

- HTML
- JS
- CSS
- images

It does not need a container server unless you are doing server-side rendering.

That is why the natural AWS mapping is:

- S3 stores the files
- CloudFront serves them globally

The design intent:

- cheap storage
- edge caching
- simple deploy
- easy cache control

This is also why the old local pattern:

- React served by Nginx

often disappears in AWS.

Nginx was mostly doing:

- static file serving
- maybe reverse proxy

CloudFront and API Gateway / ALB can absorb those jobs.

## 13. Why Databases Are Separated From App Compute

In local Docker Compose, it is normal to run:

- app
- postgres
- redis

on one machine.

In production, that is usually the wrong durability boundary.

The database wants:

- durable storage
- backups
- snapshots
- maintenance windows
- parameter tuning
- controlled failover

Your app wants:

- frequent deployments
- scaling
- statelessness
- easy replacement

These are different lifecycles.

RDS exists because AWS wants the data layer to be more durable and more operationally isolated than the app layer.

That is good separation.

## 14. IAM Roles Instead of AWS Keys

This is worth saying again because it changes almost everything.

For AWS service access:

- Lambda should use its execution role
- ECS tasks should use a task role
- EC2 should use an instance profile
- GitHub Actions should assume a role via OIDC

That means:

- no access key in env vars
- no S3 key in Secrets Manager
- no "shared deploy user" if you can avoid it

This is one of the main design intents of modern AWS setups:

- make cloud access tied to identity and trust
- not long-lived static credentials

## 15. The Subtle Exception: App Secrets Still Exist

Saying "use IAM instead of secrets" can become misleading if taken too literally.

IAM does not replace every secret on Earth.

It replaces many **AWS access** secrets.

But your app may still need:

- third-party API tokens
- OAuth secrets
- webhook signing keys
- SMTP creds

Those still belong in:

- Secrets Manager
- SSM Parameter Store
- or some company-approved secret system

So the more precise statement is:

> Use IAM roles for cloud permissions.  
> Use a secret store for actual application secrets.

That distinction makes design conversations much cleaner.

## 16. Postgres Without a Stored App Password

This is where a lot of people understandably get stuck.

The question is:

"If my app should not have a DB password sitting around, how does it connect to Postgres?"

On AWS with RDS PostgreSQL, one strong answer is:

- enable IAM database authentication
- create a PostgreSQL user for the app
- grant that user the `rds_iam` capability
- give the app's IAM role permission to connect as that DB user
- generate a short-lived auth token at runtime

So the app still has:

- host
- port
- database name
- database user name

But instead of a long-lived password, it generates a temporary token.

This is a beautiful design because it aligns DB access with workload identity.

Important nuance:

- the DB user still exists inside PostgreSQL
- IAM does not replace PostgreSQL's permission model

Think of it as two layers:

### 16.1. AWS IAM layer

Who is allowed to attempt a DB connection as this username?

### 16.2. PostgreSQL role layer

What is that DB user allowed to do once connected?

This is a great example of how identity systems stack rather than replace each other.

## 17. Why the Master DB User Should Not Be Your App User

This is one of those things that seems obvious after you hear it once.

The database admin user exists for:

- bootstrap
- maintenance
- migrations
- user creation

Your application user should be narrower:

- read/write on specific schemas and tables
- not superuser-like
- not able to do everything

That means a good pattern is:

- RDS manages the master/admin password
- bootstrap creates app-specific DB users
- app connects as a least-privileged user

That is better security and better operational discipline.

## 18. Why the Browser Usually Should Not Talk to AWS Directly

This is another common source of confusion.

Your React app running in the browser does not naturally have an AWS IAM role.

The browser is not your backend.

So the usual pattern is:

- browser talks to your API
- API, using its own IAM role, talks to AWS services

For uploads, a very common flow is:

1. browser asks API for an upload URL
2. API generates a presigned S3 URL
3. browser uploads directly to S3

This is elegant because:

- browser never gets broad AWS permissions
- upload traffic bypasses your API server
- your backend stays in control of authorization

There are ways to give browser-side AWS identities, but for a normal app this is usually unnecessary complexity.

## 19. Background Jobs Change the Architecture

As soon as you say:

- "I have a runner"
- "it polls the DB"
- "it launches Docker jobs"

you are no longer talking about just a web app.

You now have:

- a control plane
- one or more execution planes

This is a major conceptual shift.

### 19.1. Control plane

The thing deciding what should run next.

In your case:

- a runner service
- polls Postgres
- claims jobs
- launches job containers

### 19.2. Execution plane

The thing that actually performs the job.

In your case:

- Docker images
- run per job
- similar to a GitHub Actions runner model

This split matters because the two halves have different needs:

- the runner is long-lived and coordination-heavy
- the worker is ephemeral and execution-heavy

That maps very nicely to AWS once you stop treating them as one thing.

## 20. How a Runner / Worker System Maps to AWS

A natural mapping is:

- runner -> ECS service
- worker jobs -> AWS Batch or ECS RunTask
- images -> ECR
- state / queue metadata -> Postgres

Why this is clean:

- runner stays alive and keeps polling
- workers are launched on demand
- workers are isolated from each other
- IAM can differ between runner and worker
- failures are easier to reason about

This is the deeper pattern:

- **service-style components live as services**
- **job-style components run as jobs**

Trying to cram both into one runtime model leads to pain.

## 21. Why AWS Batch Often Fits Better Than DIY Worker Launching

If your jobs are:

- one-shot
- containerized
- sometimes resource-heavy
- independent from each other

AWS Batch is often the right abstraction.

Batch gives you:

- job queues
- job definitions
- retries
- scheduling semantics
- compute environment separation

This is a better model than:

- manually SSH into a box
- manually run Docker containers
- hope cleanup works

Even if your runner is custom, AWS Batch can still be the execution layer.

## 22. Why Fargate Is Great Until It Isn't

Fargate is appealing because:

- no host management
- simple container deployment
- clean IAM integration
- nice service model

But it has a design center:

- ordinary container workloads
- not full host-level control

So if your job workers need:

- privileged containers
- Docker-in-Docker
- special kernel behavior
- host socket access

you may need EC2-backed ECS or EC2-backed Batch instead.

That is not AWS being bad.

That is just a reminder that "serverless containers" still come with guardrails.

## 23. Networking: Why So Much of AWS Feels Like Plumbing

AWS networking feels annoying until you realize what it is protecting you from.

The platform wants you to say explicitly:

- which things are public
- which things are private
- which subnets a workload lives in
- which port flows are allowed

That is why you keep seeing:

- private subnets for app and DB
- public load balancers or public edge endpoints
- security groups limiting ingress

This is one of the strongest design instincts in cloud architecture:

- default toward private
- make public exposure explicit

That instinct is almost always correct.

## 24. Security Groups: The Practical Mental Model

Security groups are easiest to think of as:

- stateful allow-lists around workloads

Instead of:

- "open port 5432 to the world"

you want:

- "allow ECS tasks in the API security group to talk to RDS on 5432"

That is a much healthier expression of intent.

The goal is not just security theater. It is:

- keep blast radius small
- make topology explicit
- reduce accidental exposure

## 25. Logging and Observability Are Part of the Architecture

A lot of beginners think of logging as an afterthought.

In AWS, logging is part of the runtime contract.

A workload that cannot emit logs and metrics is barely deployable.

So the practical model is:

- every compute unit should have a logging path
- every deploy should make debugging easier, not harder

That usually means:

- CloudWatch Logs for app logs
- structured logging in your app
- metrics and alarms where it matters

Terraform often creates the resources and permissions for this, even if the app emits the actual data.

## 26. Drift: Why Manual Console Changes Cause Trouble

Drift means:

- the real infrastructure no longer matches Terraform's declared configuration

This happens when:

- somebody clicks in the console
- somebody runs a one-off CLI mutation
- a provider or managed service changes behavior

Drift is dangerous because it breaks the social contract:

- the code is no longer the whole truth

That is why teams get grumpy about "just this one manual change."

It is not purism. It is the cost of losing reliable infrastructure intent.

## 27. Why You Still Need the Console

Even strong Terraform teams still use the AWS console constantly.

Just not as the authoring surface.

You use the console to:

- inspect task logs
- read RDS events
- inspect CloudFront behavior
- verify security groups
- understand a live incident

So the healthy attitude is:

- Terraform for creation and change management
- console for visibility and live operations

Not:

- never touch the console

That becomes dogma and is not useful.

## 28. How a Monorepo Changes the Deployment Conversation

Once everything lives in one GitHub monorepo, deployment becomes less about "the app" and more about **surfaces of change**.

You may have:

- frontend code
- API code
- runner code
- worker images
- Terraform infra code

Those do not all need the same pipeline.

A monorepo pushes you toward:

- path-based workflow triggers
- environment-aware deployments
- shared versioning through commit SHA

That is a good fit for Terraform because Terraform also wants explicit structure and environment separation.

## 29. How CI/CD Fits Into This Model

The clean mental model is:

### 29.1. Build pipeline

Turns source code into deployable artifacts:

- frontend build output
- API image
- runner image
- worker images

### 29.2. Infra pipeline

Ensures the platform exists:

- VPC
- ECS cluster
- Batch queue
- ECR repos
- S3 buckets
- CloudFront
- RDS
- IAM roles

### 29.3. Deploy pipeline

Connects new artifacts to existing infrastructure:

- update ECS services to new image versions
- update Batch job definitions
- upload frontend assets
- run DB migrations

If you blur these three together, deploy systems get messy fast.

## 30. GitHub Actions and OIDC: Why This Is Better Than Stored AWS Keys

If GitHub Actions is deploying to AWS, the old pattern was:

- create a long-lived AWS access key
- store it in GitHub secrets

This works, but it is weak:

- the key exists until rotated
- compromise window is long
- permissions often become too broad

The better pattern is:

- GitHub emits an OIDC identity token for the workflow
- AWS IAM trusts that token source under specific conditions
- the workflow assumes an AWS role
- AWS issues short-lived credentials for that run

This is conceptually the same security move as using roles for workloads.

Identity over static credentials.

Once you see that, it feels much more coherent.

## 31. The Difference Between "Provision" and "Deploy"

This vocabulary matters.

### Provision

Create or update infrastructure resources.

Examples:

- create an RDS instance
- create an ECS cluster
- create an S3 bucket
- create an IAM role

### Deploy

Roll out a new version of application code onto existing infrastructure.

Examples:

- push a new API container image
- update an ECS service
- upload new frontend assets
- run a new migration

Companies often use Terraform for provisioning and something lighter for day-to-day app deploys.

Sometimes Terraform also owns deploy-time image references. Sometimes it doesn't.

The main thing is to know which layer you are touching.

## 32. Why Immutable Artifacts Matter

A deploy gets much safer when the artifact is immutable.

For containers, that usually means:

- tag by commit SHA
- ideally pin by digest

For frontend assets, that usually means:

- content-hashed file names

Why this matters:

- you know exactly what was deployed
- rollback is easier
- caches behave better
- "latest" stops causing ambiguity

This is a very practical expression of good infra design:

- reduce ambiguity
- make runtime identity concrete

## 33. Why Migrations Should Be Treated as First-Class

Databases are where "just deploy it" often goes wrong.

A good mental model is:

- schema evolution is part of the deploy contract
- not an afterthought

That means migrations should be:

- explicit
- reviewable
- versioned
- run in the same environment model as the app

For containerized systems, a clean pattern is:

- run migrations as a one-off task using the same image family and IAM/network context as production

That keeps surprises down.

## 34. The Most Useful AWS Architecture Instincts

If you are early in learning AWS, these instincts will serve you well:

### 34.1. Prefer managed services when they fit

Do not run a database in a container if RDS is the real need.

### 34.2. Prefer identity over static credentials

Use roles whenever possible.

### 34.3. Prefer private networking by default

Expose only what must be public.

### 34.4. Separate control-plane work from execution work

Long-lived coordinators and short-lived jobs are not the same thing.

### 34.5. Treat data durability and app deployability as different concerns

Databases and app containers have different lifecycle needs.

### 34.6. Keep the source of truth in Git

If the code does not describe the system, eventually the system becomes folklore.

## 35. Mapping Your Specific Stack

Based on the questions you were asking, a coherent AWS/Terraform architecture would look like this:

- React frontend served from `S3 + CloudFront`
- FastAPI API running as an `ECS/Fargate service`
- Postgres on `RDS for PostgreSQL`
- file storage in `S3`
- API uses an ECS task role for AWS access
- API uses IAM DB auth to connect to Postgres without a stored app password
- job runner as a separate `ECS/Fargate service`
- job workers launched via `AWS Batch` or `ECS RunTask`
- worker images stored in `ECR`
- Terraform owns all shared infra, roles, policies, and networking
- GitHub Actions uses OIDC to assume deploy roles

The reason this feels coherent is that each component matches the thing it actually is:

- static assets -> object storage + CDN
- HTTP API -> service
- database -> managed database
- long-lived coordinator -> service
- ephemeral worker -> batch job

That alignment is the main design win.

## 36. Why FastAPI on ECS Usually Makes Sense Here

You can absolutely run FastAPI on Lambda.

But once you also have:

- a runner
- job workers
- a monorepo built around Docker images
- a DB-backed architecture

ECS often becomes the cleaner center of gravity.

Why:

- one container packaging model
- easier shared VPC story
- normal service semantics
- cleaner fit for long-lived processes
- less glue code to make a server framework feel natural in a function runtime

This is not a universal law. It is just a very practical fit.

## 37. Where Terraform Should Stop

This is an underrated skill: knowing what **not** to force into Terraform.

Terraform is usually a poor home for:

- application business logic
- long procedural deployment scripts
- rich data migrations
- job claiming logic
- request routing logic inside the app
- normal software module boundaries

Terraform shines when describing:

- resources
- permissions
- wiring
- environment shape

Trying to stretch it far past that usually makes the code worse.

## 38. Common Beginner Confusions

These are extremely normal.

### 38.1. "Why do I need Terraform if AWS already has a console?"

Because the console is not a versioned, reviewable, reproducible source of truth.

### 38.2. "Why can't my frontend just use the backend's IAM role?"

Because the browser is not your backend runtime and should not inherit broad cloud permissions.

### 38.3. "If I use IAM, why do I still have a DB user?"

Because IAM controls cloud identity, while PostgreSQL still controls database privileges.

### 38.4. "Why can't Terraform just deploy everything?"

It can own a lot, but runtime deploy orchestration and infra provisioning are different jobs.

### 38.5. "Why do I still need the console if Terraform is the source of truth?"

Because visibility and authoring are different concerns.

## 39. Good Questions to Ask in a New Company Terraform Repo

If you are entering a Terraform-heavy company, these questions cut through a lot of confusion:

- Where is Terraform state stored?
- How are environments separated?
- Which modules are shared vs app-specific?
- Does Terraform own image versions, or does app deploy update services separately?
- How are secrets handled?
- Do services use IAM DB auth or Secrets Manager-backed DB creds?
- What is the standard deploy path for frontend, API, and workers?
- What is the rollback story?
- Which resources are allowed to be changed manually in emergencies?

These questions get you from "where do I click?" to "how does this platform think?"

## 40. The Design Intent of the Whole System

This is the part to internalize.

Good cloud architecture is not about collecting services.

It is about making these properties true:

- infrastructure is reproducible
- permissions are explicit
- credentials are short-lived when possible
- public exposure is deliberate
- stateful data is treated carefully
- compute can be replaced safely
- deploys are predictable
- failures are observable

Terraform supports that by making the infrastructure itself:

- textual
- reviewable
- versioned
- diffable

AWS supports that by giving you:

- managed service boundaries
- identity primitives
- network isolation
- durable storage systems
- multiple compute models

Once you stop seeing AWS as "a million random products" and Terraform as "a strange config language," the whole picture gets simpler.

It becomes:

- design the boundaries well
- choose the right runtime for each job
- let IAM carry identity
- let Terraform describe the platform
- let CI/CD move artifacts through it safely

That is the real crash course.

## 41. A Short Practical Summary

If I had to compress all of this into one page:

- Terraform is for provisioning and describing infrastructure, not for being your application runtime.
- AWS works best when you separate identity, networking, compute, data, and edge delivery in your head.
- IAM roles should replace most AWS access keys.
- App secrets still exist, but cloud permissions should usually ride on workload identity.
- Static frontends usually belong in S3 + CloudFront.
- APIs and runners usually belong in Lambda or ECS, depending on how long-lived and container-shaped they are.
- Databases belong in RDS, with app-specific users and narrow privileges.
- Background job systems naturally split into a control plane and execution plane.
- GitHub Actions should usually assume AWS roles with OIDC, not use long-lived access keys.
- Good infra design is mostly about clear boundaries and reducing ambiguity.

If you feel overwhelmed, that is normal.

The trick is not to memorize services.

The trick is to keep asking:

- What is this component actually responsible for?
- What identity should it have?
- What should it be allowed to touch?
- Is it long-lived or ephemeral?
- Is it stateful or stateless?
- Is this provisioning or runtime behavior?

Those questions will get you surprisingly far.
