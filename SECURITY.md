# Security

Do not report or commit production credentials, private model payloads, or proprietary market datasets in this public CI mirror.

If a credential is accidentally committed, revoke it immediately and remove it from history before any further public use.

Production secrets belong only in the deployment platform secret manager. Production model/data artifacts are mounted at runtime and are intentionally absent from this repository.
