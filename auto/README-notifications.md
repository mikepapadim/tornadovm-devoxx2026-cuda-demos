# ntfy monitoring

The autonomous study can send lightweight phone notifications through [ntfy.sh](https://ntfy.sh/). ntfy topics are public by default, so use a long, unguessable topic name; if confidentiality matters, use an authenticated/protected topic and set `NTFY_TOKEN` rather than putting credentials in the repository.

## 1. Subscribe on your phone

Install the ntfy app and subscribe to your chosen topic. The ntfy web app also supports subscriptions. See the official ntfy publishing/subscription documentation.

## 2. Test from the server

```bash
export NTFY_TOPIC='choose-a-long-random-topic'
./auto/notify.sh 'TornadoVM study notification test'
```

If you use an access token:

```bash
export NTFY_TOKEN='tk_...'
```

Do not commit the topic or token to Git if you want them private. A random, hard-to-guess topic is especially important on the public ntfy.sh service.

## 3. Supervisor configuration

The supervisor accepts `NOTIFY_URL` for backward-compatible simple webhook notifications. For ntfy, configure the full publish URL:

```bash
export NOTIFY_URL="https://ntfy.sh/$NTFY_TOPIC"
```

For protected topics, use the token-aware helper or configure the supervisor's notification environment so credentials remain outside Git.

## 4. Recommended notifications

The supervisor should notify on:

- supervisor start/stop;
- task completion/progress;
- task blocked after repeated stalls;
- Claude authentication errors;
- rate-limit pauses when material;
- profiler/GPU blockers;
- queue completion.

Avoid sending every shell command or profiler line. Notifications are for exceptions and meaningful milestones; detailed logs stay in `auto/logs/` and Git.
