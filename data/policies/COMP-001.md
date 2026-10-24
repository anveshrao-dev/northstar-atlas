# COMP-001 — Access Review & Audit Evidence Policy

**Organization:** Northstar Technologies  
**Document ID:** COMP-001  
**Department:** Compliance  
**Policy Owner:** Head of Compliance (proposed)  
**Version:** 0.1  
**Status:** Draft — Proposed (Synthetic)  
**Effective Date:** Pending approval  
**Review Cycle:** Proposed annual review  
**Classification:** Internal  
**Language:** English  
**Draft Date:** 2026-10-21

## 1. Purpose and Scope

### 1.1 Internal Evidence Governance

This policy proposes how the fictional Northstar Technologies maintains,
reviews, protects, provides, and disposes of evidence for access administration,
employee lifecycle events, production changes, and incidents across cloud and
on-premises systems. It covers records produced by HR, IAM, Cloud Platform,
Data Center, database, application, Security, SRE, and Change Management teams.

Evidence must distinguish what was requested, who authorized it, what actually
happened, and how the result was verified. Possession of evidence or responsibility
for its retention does not authorize the underlying access or technical action.
This is an internal governance proposal, not a claim of regulatory compliance,
audit certification, a legal opinion, or a real company's recordkeeping practice.

### 1.2 Draft Authority

**All COMP-specific requirements are proposed and not effective until reviewed
and approved.** “Must” denotes a proposed requirement. HR-001 v1.0 remains
Approved (Synthetic) and authoritative for employee lifecycle rules. IT-001,
PLAT-001, SRE-001, and SEC-001 v0.1 remain Draft — Proposed (Synthetic).
Accepted project deliverables do not become effective policies through citation.

This document adds evidence-governance proposals without changing those sources,
their approval boundaries, or HR deadlines. It defines no regulatory retention
period or universal review interval. Actual schedules, authorities, and technical
mechanisms must be agreed before dependent processes are used operationally.

## 2. Roles and Responsibilities

The following ownership model is proposed.

| Role | Evidence responsibility and boundary |
| --- | --- |
| Compliance | Own evidence governance, retention standards, access-review requirements, completeness oversight, and coordination of audit requests; do not approve operational actions merely by accepting their evidence |
| IT Operations / IAM | Own workforce identity/access administration records, lifecycle access coordination evidence, review inventories, and central delegation records; reconcile specialist execution results |
| Platform Engineering | Own cloud infrastructure, cloud authorization implementation, and deployment execution evidence, including Terraform plans and technical-review records; do not replace Change Management authorization |
| Data Center Infrastructure teams | Own records for their on-premises Windows/AD infrastructure, Linux, Jenkins, networking, and identity integrations; enterprise workforce governance remains with IAM |
| Database teams | Own database permission, administration, change, and verification evidence; link underlying infrastructure dependencies |
| Information Security | Own security decisions, risk/classification records, privileged-access control evidence, and security exception records; technical owners contribute implementation and verification evidence |
| SRE | Own incident coordination records, timelines, communication, restoration evidence coordination, closure, and review/action tracking; technical teams retain evidence for their own interventions |
| Change Management | Own production change authorization records, approved scope/window/conditions, and amendment decisions; do not technically approve Terraform plans or mappings |
| Human Resources | Own employee lifecycle records and notification evidence under HR-001; do not own every system's access-removal log |
| Managers and application/service owners | Review continuing business need, supply applicable application approvals, and own application execution/validation evidence |
| Evidence custodians and authorized reviewers | Operate repositories or examine records within documented access and scope; custody or reviewer status grants no broader production authority |

Each evidence category must have an accountable owner and a designated repository
or source system. Delegated collection or storage does not remove that owner's
responsibility. Shared events need linked records and explicit handoffs rather
than an assumption that another team has captured the evidence.

## 3. Evidence Classification and Ownership

### 3.1 Evidence Catalogue

**Proposed:** Compliance maintains a catalogue with each record category's owner,
source, purpose, sensitivity, authorized audiences, required fields, linked
process/policy, review requirements, retention-schedule reference, and custodian.
Examples include HR events, access decisions and fulfillment, CR/cTASK records,
Terraform review artifacts, incident timelines, and Security decisions.

The catalogue describes accountable sources; it does not require copying every
record into one repository. A controlled index may link records held by their
owners. Identify missing records and broken links explicitly; a reference without
retrievable supporting evidence is not proof of completion.

### 3.2 Sensitive Evidence

**Proposed:** Owners assess sensitivity with Security and Compliance. Personal
lifecycle records, privileged-access logs, security findings, and detailed
infrastructure artifacts may need access restrictions beyond an ordinary internal
summary. The policy's Internal classification does not make every underlying
record suitable for broad access.

Use sanitized summaries and controlled references to sensitive originals. Do not
collect passwords, secret values, private keys, or raw secret-bearing plan/state
content into routine evidence packages. Record what was redacted and why without
reproducing the secret. Redaction must not conceal material failures or findings;
use a restricted supporting record where needed.

## 4. Access Review Governance

### 4.1 Review Scope and Responsibilities

**Proposed:** Compliance defines review requirements and requires a documented,
risk-based review plan agreed with IAM, Security, and system owners. The plan
identifies coverage, reviewers, cadence/checkpoints, source snapshots, exclusions,
decision authority, and follow-up owners. No universal numerical cadence is set.
Periodic review does not replace event-driven action for transfers, departures,
temporary-access expiry, changes to delegation or workload purpose, or incidents.

IAM assembles and reconciles employee, workload, group, and delegation inventories.
Managers or accountable business/service owners attest continuing business need.
Security reviews sensitive/privileged scope and control conditions. Platform,
Data Center, and database owners validate effective permissions and provide
system-specific findings, including inherited access and known coverage gaps.
Compliance checks completion, evidence quality, and unresolved findings.

Reviewers must not be the sole attester of their own privileged access; use an
independent authorized reviewer. Actual reviewer assignments and alternative
approvers need documented delegation rather than assumed authority.

### 4.2 Decisions and Remediation

**Proposed:** Record the subject/workload, environment, effective permissions,
source snapshot and time, business purpose, relevant approvals, reviewer and
authority, decision, rationale, exceptions, and required actions. Outcomes may
retain justified access, request modification/removal, or require investigation.
Missing evidence remains an unresolved finding, not an automatic pass.

A review does not retroactively authorize an unapproved grant or create new
permissions. Route remediation through the applicable access/change process;
record executor, result, verification, owner, and agreed completion checkpoint.
Escalate blocked or overdue actions to the responsible owner and Security where
appropriate. Closing a review ticket alone does not establish removal, and an
agreed follow-up checkpoint does not extend HR deadlines or access expiry.

## 5. Employee Lifecycle Evidence

### 5.1 Lifecycle Records and Notifications

**Proposed:** HR retains the lifecycle event record, relevant manager approval,
effective date/time, notification content and dispatch evidence, and recorded
exceptions. IAM retains receipt/acknowledgment and the linked access work record.
Use controlled identity references to correlate records without unnecessarily
duplicating personal information in broad audit packages.

HR-001 §§4.1–4.5 govern role-change initiation, notifications, access review, new
requests, and administrative completion. Preserve the distinction between planned
and urgent events. HR-001 §§5.1–5.2 govern departure notification, including
“where practicable,” and disablement at the approved termination/separation time.
Evidence timestamps do not redefine the HR business-day calendar or deadlines.

### 5.2 Access Removal and Completion

**Proposed:** IAM owns the coordinated access-removal evidence, with Platform,
Data Center, database, and other technical owners maintaining their execution
and verification records. Link the HR event, affected access inventory, approved
effective time, technical actions and actual times, verification, failed/partial
removals, escalation, and remaining gaps. Preserve HR's lifecycle evidence and
each team's source records rather than attributing every removal to HR.

Role-change completion must show the five HR-001 §4.5 conditions; administrative
completion does not authorize privileged access. Offboarding evidence under
HR-001 §5.4 includes the relevant HR and IT records. Equipment-return records
remain separate from access disablement; an outstanding device does not extend
the termination deadline. Workload ownership changes are tracked separately under
proposed IT-001 §8.3, not treated as retention of a departing person's access.

## 6. Privileged Access Evidence

### 6.1 Decisions, Delegation, and Fulfillment

**Proposed:** Preserve the access request and business justification, subject/type,
target scope, risk classification, business and Security decisions, applicable
IAM/Security delegation, technical mapping/validation, MFA or workload control
verification, executor, actual grant and verification, and any exception.
Link production change authorization separately where applicable.

Security owns Security decisions and control evidence; IAM owns central access
and delegation records; technical teams own provision/revocation evidence.
This follows proposed IT-001 §§5.2–5.3 and 11 and SEC-001 §§2.2, 5.2, and 13.
Evidence must show which decision was made by which authorized role, not merely
that a ticket contains an approval label. Never use credentials as proof.

### 6.2 Temporary and Emergency Expiry Review

**Proposed:** Record approved temporary scope and start/end conditions, expiry
enforcement, named revocation owner, and any renewed authorization. Technical
owners execute and verify removal; IAM reconciles results across systems and
tracks incomplete removal; Security reviews privileged expiry/control evidence
and exceptions. Compliance checks that the required review and remediation
records exist and unresolved findings remain visible.

Record actual removal times, effective-permission checks, supported session/token
checks, and limitations. A scheduled expiry, closed incident, or central account
disablement is not by itself proof that every grant or session ended. Extensions
require their own decision record under proposed SEC-001 §9, not an altered
original expiry. Preserve emergency authorization and follow-up evidence without
inventing an executable emergency process or access duration.

## 7. Cloud and On-premises Access Evidence

**Proposed:** Evidence follows the hybrid ownership model in proposed SEC-001
§§2, 6, and 7. Platform supplies cloud Azure/GCP authorization, Kubernetes RBAC,
managed identity/federation, Key Vault integration, and cloud-hosted Jenkins
implementation evidence. Data Center teams supply their Windows/AD infrastructure,
Linux, on-premises Jenkins, network, and integration records. Database teams supply
database-specific permission and verification records; IAM retains workforce
identity/group governance and delegation records.

Record the hosting context, component, subject/workload, effective scope, technical
owner, delegation, action/result, and cross-system dependencies. A cloud-hosted
pipeline using an on-premises target needs evidence from both owners. Distinguish
Key Vault administration from secret retrieval, database permissions from host
access, and operating AD infrastructure from governing employee identities.

Reconcile discrepancies between inventories and actual system evidence. Record
unsupported checks and missing local-account, inherited-permission, or session
coverage; do not treat central directory evidence as proof of every system's
state. Protect sensitive references under §§3 and 12.

## 8. Production Change and Deployment Evidence

### 8.1 CR, cTASK, and Execution Chain

**Proposed:** Link business purpose, affected applications/infrastructure, change
class, implementation plans and cTASK sequence, responsible teams, dependencies,
nonproduction validation, application approvals, completed prerequisites, required
Security/readiness decisions, and final Change Management authorization.
Preserve approved scope, risk, window, conditions, and recovery bounds under
proposed PLAT-001 §§4.1–4.4 and 8. A CR opened during preparation is not proof
of final approval; a production cTASK planned before approval is not executed work.

Change Management owns production authorization and amendment records. Platform,
application, Data Center, database, and network owners retain their execution
evidence for applicable tasks: reviewed source/artifact/configuration references,
pipeline or executor identity, target, actual times, results, verification,
failures, deviations, and recovery. Link handoffs, incidents, escalation, and
authorized window decisions. A successful pipeline alone does not establish
application health or complete change evidence.

### 8.2 Terraform Technical Review and Plan Identity

**Proposed:** Platform maintains the production plan's protected source/artifact
reference and integrity identifier, source/configuration/input context, provider
and target/state context, review findings on creates/updates/replacements/deletions,
and recorded technical approval by an authorized Platform reviewer. Preserve
normal-change author/reviewer separation and proof of the plan actually applied,
executor, execution result, and infrastructure/application verification.

Connect this record to the approved CR/cTASK and separate Change Management
decision. Change Management is not the Terraform technical approver. A plan file
alone does not prove approval; a successful apply does not prove authorization.

For regenerated plans or drift, preserve the replaced and current plan references,
investigation/corrections, renewed Platform review/approval, and comparison with
authorized CR scope, risk, window, and other conditions. Proposed PLAT-001 §6.1
requires no separate Change Management approval solely for in-scope regeneration;
material scope/risk expansion or window overrun requires the applicable amendment
or escalation decision. Evidence must distinguish those cases.

Use sanitized summaries and controlled references, not raw secret-bearing plans
or Terraform state in ordinary change or audit packages. State storage and
protection remain governed by proposed PLAT-001 §6.2; audit requests follow §12.

## 9. Incident Response Evidence

**Proposed:** SRE owns the coordinated reliability incident record under proposed
SRE-001 §§9.2, 11, and 12. Retain discovery and estimated onset, declaration,
severity rationale/changes, named ownership and acknowledged handoffs, hypotheses
versus confirmed findings, actions and decisions, communications, escalations,
restoration criteria and verification, residual impact, closure rationale/time,
and post-incident review/actions with accountable owners and completion evidence.

Technical teams retain intervention and component-verification records; Security
owns security decisions and related control evidence. Link relevant CR/cTASK,
access, exception, certificate/credential, and Security response records. Rapid
Response involvement proves coordination activity, not window/access approval.

Incident closure does not delete the evidence, prove the root cause was found,
or complete every corrective action. Preserve partial-restoration findings,
workarounds, residual-risk decisions, outstanding tasks, and any linked recurrence.
Distinguish planned actions from completed and verified ones. Retention follows
the approved category schedule and hold status, not an assumption that closure
permits deletion. This policy introduces no universal review deadline.

## 10. Evidence Integrity and Traceability

### 10.1 Provenance and Protected Sources

**Proposed:** Records identify their owner/source system, stable record identifier,
related event/subject references, actor, event and collection times with time-zone
context, applicable decision authority, policy/version context, and supporting
artifact references. Mark estimated or unknown times and known collection gaps.

Preserve source versions or controlled snapshots as needed to support the recorded
decision. Use access restrictions, change history, and integrity checks appropriate
to the repository and record type. An artifact hash can distinguish versions but
does not by itself prove who approved the artifact or that its contents are true.
Protect sensitive originals; broad indexes should contain controlled references.

### 10.2 Corrections and Missing Evidence

**Proposed:** Corrections record the original reference, corrected information,
reason, author, time, and applicable review. Preserve the correction history;
do not silently overwrite material decisions, backdate approvals, or fabricate
missing execution evidence. A redacted derivative identifies its source version
and handling decision without exposing removed values.

Log missing, corrupt, contradictory, or inaccessible evidence as a finding with
owner, impact, investigation, and follow-up checkpoint. Later reconstructed
information must be labeled as such with its basis and uncertainty; it cannot
establish that an undocumented authorization existed. Escalate suspected tampering
to Security and Compliance. Repository implementation details remain to be agreed.

## 11. Retention and Disposal Governance

### 11.1 Approved Retention Schedule

**Proposed:** Compliance owns a versioned internal retention schedule, developed
with record owners, Security, and designated legal advisers where appropriate.
Each category records purpose, accountable owner, retention start event, proposed
period, rationale, source of any separately verified obligation, protected storage,
review conditions, hold handling, disposition method, and approving authority.
An authorized retention-governance approver must record adoption; the actual
approver/delegation remains to be designated.

No numeric period is set here. Future durations must be labeled proposed internal
standards until approved; legal or contractual requirements need a separately
verified source rather than inference from this synthetic policy. The annual
policy review cycle is not an evidence-retention period or access-review cadence.

Where a schedule is missing or unclear, record the gap and obtain a documented
interim preservation/disposition decision from authorized governance with owner
and Security input, purpose, scope, and a review checkpoint before disposal.
Uncertainty is neither automatic permission to delete nor a mandate for indefinite
retention. Do not silently select a period or claim that all earlier retention
dependencies are resolved by the existence of COMP-001.

### 11.2 Controlled Secure Disposal

**Proposed:** Before disposal, the custodian verifies the applicable approved
schedule/version, trigger and eligibility, outstanding holds or preservation
requests, relevant owner confirmation, and authorized disposition decision.
Use a Security-reviewed method appropriate to the medium and repository, covering
managed exports, replicas, and backup lifecycle where applicable. Record copies
that cannot yet be removed and their protected handling; do not claim complete
erasure without evidence.

Maintain a disposal record identifying categories/record references, authority,
method, executor, time, result, verification, and exceptions without retaining
the deleted sensitive payload. That disposal record needs its own schedule.
Release of a hold only permits reassessment under the schedule; it is not a
command to delete. No actual deletion or disposal technology is implemented here.

## 12. Audit Access and Evidence Requests

### 12.1 Scoped Requests and Release Decisions

**Proposed:** Compliance coordinates requests identifying requester identity and
authority, purpose, record categories, scope/period, recipients, intended use,
transfer/access method, and required completion checkpoint. Owners identify
responsive sources; Security reviews sensitive access and release conditions.
Record the authorized release decision, recipient access scope, material supplied,
redactions, limitations, and access withdrawal/return/disposal conditions.

Auditor status or an evidence request does not automatically grant production
access, broad repository access, or permission to receive secrets. Requests outside
authority or with unclear scope require clarification. Provision any approved
review access through the applicable IAM/Security process; record fulfillment
and expiry verification separately from the request.

### 12.2 Terraform State and Other Sensitive Originals

**Proposed:** An auditor may request evidence, but a request for raw Terraform
state containing secrets is not authorization to disclose it. Start with a
sanitized plan/change summary, approval and execution records, and protected
artifact references sufficient to address the audit purpose. Never copy raw
secret-bearing state into ordinary tickets, email attachments, or audit packages.

If those materials are insufficient, Compliance, the source owner, and Security
must assess necessity and authorize a suitably restricted/redacted inspection
method, recipients, and logging before access. This policy grants no standing
entitlement to raw secrets or approval for a particular original artifact release.
If safe and authorized access cannot be established, record the limitation and
escalate the request rather than expose credentials or invent substitute evidence.

## 13. Exceptions and Legal Holds

### 13.1 Bounded Evidence Exceptions

**Proposed:** Record the requirement, reason, scope, risk, compensating controls,
owner, expiry, approving authority, decision, and review/expiry action. The Head
of Compliance may approve only evidence-governance exceptions within documented
authority, with affected owners and required Security concurrence.

An evidence exception cannot retroactively authorize access or deployment, waive
another owner's control, extend HR termination deadlines, release a legal hold,
or silently extend retention or access. Missing evidence remains disclosed even
when its remediation approach is accepted. Actual delegation limits and exception
procedures require approval before operational use.

### 13.2 Preservation Requests and Hold Lifecycle

**Proposed internal process:** Route a potential hold or preservation request to
Compliance and the designated legal-hold decision-maker for assessment. Record
the request, scope, sources/custodians, receipt time, and responsible assessor;
protect potentially affected records from scheduled disposal pending that recorded
assessment. Track the request to resolution rather than leave it unowned.

A hold record must identify issuing authority, reason/reference, covered records
and copies, custodians, preservation instructions, acknowledgments, restrictions,
review responsibility, and any changes. Custodians suspend disposal for covered
material and verify preservation while maintaining access and Security controls.
Report gaps or inability to preserve to Compliance and the issuing authority.
Preserving a record does not authorize continued operational access by its subject.

Only the designated authorized hold decision-maker may direct a recorded release
or scope change. Routine expiry, incident closure, or a Compliance exception does
not release a hold. After release, reassess disposition under §11 and record it.
Actual hold authority and procedures require governance/legal review; this section
asserts no statutory trigger, jurisdiction, duration, or legal obligation for the
fictional organization and establishes no current hold.

## 14. Policy Ownership and Review

**Proposed:** The Head of Compliance owns this policy and coordinates review with
HR, IT/IAM, Platform, Data Center, database/application owners, Security, SRE,
Change Management, and designated legal advisers where appropriate. Review is
annual and triggered by material process, evidence-system, incident, or
companion-policy changes.

An authorized governance approver approves publication after required reviews;
the actual authority remains to be confirmed. Assign an effective date only
after approval. Schedule adoption, operational delegation, individual evidence
release, and legal-hold decisions remain separate from policy publication.
Version 0.1 remains Draft — Proposed (Synthetic); no committee, certification,
or existing publication approval is asserted.

## 15. Related Policies

| Document | Corpus status and relationship |
| --- | --- |
| HR-001 — Employee Role Change & Offboarding Policy | Present; v1.0, Approved (Synthetic). Authority for employee lifecycle records, notification/removal deadlines, and administrative completion |
| IT-001 — Identity & Access Management Policy | Present; v0.1, Draft — Proposed (Synthetic). Identity/access decisions, delegation and workload records, technical fulfillment, and lifecycle coordination |
| PLAT-001 — Production Deployment & Change Management Policy | Present; v0.1, Draft — Proposed (Synthetic). CR/cTASK governance, Platform technical review, Change Management authorization, execution, and recovery evidence |
| SRE-001 — Incident Response & Escalation Policy | Present; v0.1, Draft — Proposed (Synthetic). Incident coordination, timelines, restoration/closure, and post-incident review evidence |
| SEC-001 — Privileged Access & MFA Policy | Present; v0.1, Draft — Proposed (Synthetic). Hybrid ownership, Security decisions, privileged/MFA controls, expiry/removal, credentials, PKI, and sensitive evidence requirements |

**End of Document — COMP-001 v0.1 — Draft (Synthetic)**
