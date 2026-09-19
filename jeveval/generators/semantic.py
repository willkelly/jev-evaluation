"""Support-ticket routing: the positive control.

Every other generator in this package puts the model somewhere it was probably
never trained -- 3SAT, reachability, Sudoku, Dyck words. When a result there is
bad, three explanations fit it equally well: the domain is hard, the harness is
broken, or the model is weak. This module exists to separate them. It produces a
task that is semantic, easy and unambiguous, so that a bad result *here* has only
one reading, which is that something upstream of the model is wrong. The plan's
Phase 0 smoke test and its Phase 1 calibration gate both read this generator; if
the control fails, the run stops.

The task is routing a support ticket to one of 8 departments. Random guessing on
the choice question is therefore 0.125, and because the class design is balanced
(each block of 8 consecutive instances contains each department exactly once) the
majority-class baseline is 0.125 as well. The noul question is balanced yes/no,
baseline 0.5. The score question is 4 ordered urgency levels, baseline 0.25.

Ground truth is by construction. A ticket is rendered from exactly one template;
that template fixes both the department and the urgency level, and the rendered
text never contains a request that belongs anywhere else. Two mechanical
invariants hold this up, both asserted by the self-check at the bottom of the
file:

  * a template's subject, request and detail sentences contain keywords of its
    own department and of no other department;
  * the shared neutral padding sentences, the greetings, the closers and the
    shared urgency sentences contain no department keyword at all.

The only sentences that mention a second department are the distractor sentences
used at the "hard" level, and each of those explicitly rules that department out
("... so this is not a billing question"). A person still routes the ticket
correctly. A keyword baseline does not, which is the point of that level.

All three question types are offered over the same tickets: the ticket depends on
(level, padding, seed, index) and deliberately not on the question type, so
`generate_noul`, `generate_choice` and `generate_score` with the same seed return
the same states. The smoke test needs all three to round-trip, and E1 and E3
compare answers across question types on identical states.

The difficulty ladder is clean -> noisy -> hard. The default stays clean: the
control's job is to be a tripwire, and a tripwire set at an interesting height is
not a tripwire. The harder rungs exist so the control is a curve rather than a
point, and so that the cheap keyword baseline falls off while comprehension does
not -- which is the comparison that makes the control informative rather than
merely reassuring.

E3, E4 and E8 reuse this as their base task, so ticket rendering is a separate
parameterised function rather than an internal detail: `ticket_for()` and
`render_ticket()` take a number of padding sentences and where to put them, and
`Ticket.as_state()` returns the object form that E4 compares against its own
stringified equivalent.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from ..instances import Instance, choice, noul, rng_for, score, shuffled_options

NAME = "semantic"

# --------------------------------------------------------------------------
# Departments, urgency rubric, levels
# --------------------------------------------------------------------------

DEPARTMENT_LABELS: dict[str, str] = {
    "billing": "Billing",
    "shipping": "Shipping",
    "technical": "Technical Support",
    "account_access": "Account Access",
    "sales": "Sales",
    "privacy": "Privacy and Data Protection",
    "careers": "Recruiting",
    "press": "Press and Media",
}
DEPARTMENTS: list[str] = list(DEPARTMENT_LABELS)

# The rubric labels state the criterion rather than naming a vibe, because the
# correct point has to follow from the ticket text for a reader who has never
# seen this rubric before. Each template's urgency sentence is written to match
# exactly one of these.
SEVERITY_RUBRIC: list[dict[str, str]] = [
    {"id": "low", "label": "Low -- the customer says it can wait; nothing is blocked."},
    {"id": "normal", "label": "Normal -- handle in the ordinary queue within a few working days."},
    {"id": "high", "label": "High -- the customer is blocked and asks for resolution the same day."},
    {"id": "critical", "label": "Critical -- live harm or an emergency; needs attention within the hour."},
]
SEVERITIES: list[str] = [r["id"] for r in SEVERITY_RUBRIC]

LEVELS: tuple[str, ...] = ("clean", "noisy", "hard")
QUESTION_TYPES: tuple[str, ...] = ("noul", "choice", "score", "all")
PADDING_POSITIONS: tuple[str, ...] = ("head", "tail", "split")

DEFAULT_DIFFICULTY: dict[str, Any] = {"level": "clean", "question_type": "choice"}

RANDOM_BASELINE_CHOICE = 1.0 / len(DEPARTMENTS)
RANDOM_BASELINE_NOUL = 0.5
RANDOM_BASELINE_SCORE = 1.0 / len(SEVERITIES)

# --------------------------------------------------------------------------
# Department vocabulary
# --------------------------------------------------------------------------
#
# This serves two purposes. It is the cheap deterministic heuristic the scoring
# rubric asks every experiment to report against -- argmax over per-department
# keyword hits -- and it is the thing the self-check uses to prove that no
# template sentence carries another department's vocabulary. Matching is
# case-insensitive and word-bounded, with an optional trailing "s"; irregular
# forms are listed out rather than guessed at.

DEPARTMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": (
        "invoice", "invoiced", "billing", "billed", "charge", "charged",
        "charging", "refund", "refunded", "payment", "card", "receipt",
        "subscription",
    ),
    "shipping": (
        "shipment", "shipped", "shipping", "delivery", "deliveries",
        "delivered", "courier", "parcel", "tracking", "depot", "warehouse",
    ),
    "technical": (
        "error", "crash", "crashes", "crashed", "bug", "sync", "syncing",
        "api", "endpoint", "timeout", "stack trace", "csv export", "log file",
    ),
    "account_access": (
        "password", "login", "log in", "logged in", "sign in", "signed in",
        "sign-in", "locked out", "two-factor", "authenticator", "credentials",
        "sso", "session", "recovery email", "unlock",
    ),
    "sales": (
        "quote", "pricing", "procurement", "purchase order", "order form",
        "seat", "contract", "evaluate", "evaluating", "vendor", "demo",
    ),
    "privacy": (
        "personal data", "data protection", "data request", "data subject",
        "subject access", "erasure", "gdpr", "retention", "privacy",
    ),
    "careers": (
        "job", "application", "applied", "applying", "recruiter", "recruiting",
        "interview", "cv", "candidate", "hiring", "careers", "position",
    ),
    "press": (
        "journalist", "press", "media", "publication", "publishing",
        "reporter", "newsroom", "editor", "embargo", "on the record", "story",
        "piece", "statement",
    ),
}


def _keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    # Longest first so "purchase order" wins over a shorter overlapping entry.
    ordered = sorted(keywords, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in ordered) + r")s?\b", re.IGNORECASE)


_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    d: _keyword_pattern(kws) for d, kws in DEPARTMENT_KEYWORDS.items()
}


def keyword_counts(text: str) -> dict[str, int]:
    """Per-department keyword hit counts for one ticket.

    These go into `Instance.meta` so the cheap-heuristic baseline can be computed
    offline from the log without re-rendering anything. They are a baseline
    feature and never ground truth; ground truth comes from the template.
    """
    return {d: len(p.findall(text)) for d, p in _KEYWORD_PATTERNS.items()}


def keyword_baseline(counts: dict[str, int]) -> str | None:
    """Argmax over `keyword_counts`, ties broken by `DEPARTMENTS` order.

    Returns None when the ticket contains no departmental vocabulary at all,
    which an offline baseline should score as a miss rather than as an abstention
    -- a heuristic that declines to answer is still wrong.
    """
    best = max(counts.values(), default=0)
    if best == 0:
        return None
    for d in DEPARTMENTS:
        if counts[d] == best:
            return d
    return None


# --------------------------------------------------------------------------
# Shared sentence pools
# --------------------------------------------------------------------------
#
# Urgency is shared across departments rather than written per template, so that
# the score question is answered from the same evidence wherever it appears and a
# "high" billing ticket and a "high" press ticket say the same thing about time.
# None of these sentences contains department vocabulary.

URGENCY_SENTENCES: dict[str, tuple[str, ...]] = {
    "low": (
        "There is no rush at all on this one.",
        "This is not blocking anything, so whenever someone has time is fine.",
        "Next week would be completely fine for a reply.",
        "Nothing here is time sensitive; please treat it as background.",
        "I am only asking so I can plan ahead, so please take your time.",
    ),
    "normal": (
        "Sometime in the next few working days would be fine.",
        "This is not blocking me today, but I would like it sorted this week.",
        "Normal turnaround is fine, as long as it is dealt with this week.",
        "No emergency, but I would rather it did not sit for weeks.",
        "A reply within a couple of working days would be ideal.",
    ),
    "high": (
        "I am blocked until this is sorted and I need it resolved today.",
        "This is stopping me from working right now, so it has to be fixed before the end of the day.",
        "I need this resolved today; tomorrow is already too late for me.",
        "Please treat this as same-day: I cannot carry on until it is done.",
        "I am stuck on this and need it done before close of business today.",
    ),
    "critical": (
        "This is happening right now and getting worse by the hour, so I need someone on it immediately.",
        "We need someone on this within the hour; it is an emergency at our end.",
        "Please escalate this immediately -- every minute it continues makes it worse.",
        "This cannot wait even a few hours; it is actively causing harm.",
        "I am asking for immediate help: the situation is live and still running.",
    ),
}

# Padding. Neutral in both senses: no department vocabulary, and no statement
# about time, so adding padding changes neither the routing answer nor the
# urgency answer. That is what makes the length knob safe for E4.
NEUTRAL_SENTENCES: tuple[str, ...] = (
    "I have been a customer since {year}.",
    "The best email to reach me on is {email}.",
    "I am based in {city}, so there may be a few hours between replies.",
    "Let me know if you need anything else from my side.",
    "I am happy to provide any details that would help.",
    "Apologies if this has landed in the wrong inbox.",
    "I looked through your help centre first and could not find this.",
    "My own reference for this is {own_ref}, if that helps you find it.",
    "I am copying my colleague {teammate} so they can follow along.",
    "I work at {company}, in case that matters for your records.",
    "I am not sure which team handles this, so I am writing to the general inbox.",
    "Please reply by email rather than calling.",
    "Thank you for the quick replies on my previous messages.",
    "I will be travelling on Thursday but will still be reading email.",
    "I have tried to keep this short.",
    "For what it is worth, everything else has been fine for us.",
    "I am writing from my work email rather than the personal one.",
    "There is no need to send me a survey afterwards, though I will fill one in.",
    "I read the notes you sent last time and they were clear.",
    "My colleague {teammate} may write in about the same thing.",
    "If it is easier to pick this up by phone, my number is on file.",
    "I appreciate that you are probably busy.",
    "This is the first time I have written in.",
    "I have set out the details below as fully as I can.",
)

# Used only at the "hard" level. Each one names a department other than the
# ticket's own and explicitly rules it out, so the ticket stays unambiguous for a
# reader while the keyword baseline is pulled towards the wrong answer.
DISTRACTOR_SENTENCES: dict[str, tuple[str, ...]] = {
    "billing": (
        "My last invoice was correct and the card payment went through fine, so this is not a billing question.",
        "I have no problem with my invoices and no charge on the card is in dispute.",
    ),
    "shipping": (
        "The parcel from my last order was delivered on time and the tracking was accurate, so delivery is not the issue.",
        "I have no complaint about the courier; every shipment so far has arrived when it was meant to.",
    ),
    "technical": (
        "The app has not thrown a single error or crash in months, so this is not a bug report.",
        "Nothing is broken technically: no error, no crash and no sync trouble at all.",
    ),
    "account_access": (
        "I can sign in with my password and two-factor code without any trouble, so this is not a login problem.",
        "My login works, my password is current and the authenticator is behaving itself.",
    ),
    "sales": (
        "I am not asking for a quote and we are not adding a seat to the contract.",
        "This has nothing to do with procurement, pricing or a purchase order.",
    ),
    "privacy": (
        "I have no questions about how you hold my personal data and I am not making a data request.",
        "This is not about data protection or retention; I am satisfied with both.",
    ),
    "careers": (
        "I am not writing about a job, an application or an interview.",
        "This has nothing to do with recruiting or any position you have advertised.",
    ),
    "press": (
        "I am not a journalist and this is not a press or media request.",
        "No publication is involved and I am not writing a story about anyone.",
    ),
}

GREETINGS: tuple[str, ...] = (
    "Hello,",
    "Hi,",
    "Hi there,",
    "Good morning,",
    "Hello support team,",
    "Dear support team,",
)

CLOSERS: tuple[str, ...] = (
    "Thanks in advance,",
    "Many thanks,",
    "Thank you for your help,",
    "Best regards,",
    "Kind regards,",
    "Grateful for any help you can give,",
)

# At the "hard" level the subject line is either missing or says nothing, so the
# routing signal has to be read out of the body.
VAGUE_SUBJECTS: tuple[str, ...] = (
    "Question",
    "Help please",
    "Hello",
    "A query",
    "Follow-up",
    "One more thing",
    "Need some help",
    "Hi again",
)

# --------------------------------------------------------------------------
# Field pools
# --------------------------------------------------------------------------
#
# Names, companies and outlets are invented, and email addresses use the
# reserved .example TLD, so nothing generated here can be mistaken for or
# delivered to a real person.

FIRST_NAMES: tuple[str, ...] = (
    "Maya", "Tomas", "Priya", "Ingrid", "Kofi", "Lena", "Diego", "Yusuf",
    "Nora", "Hiroshi", "Aisha", "Piotr", "Mireille", "Samuel", "Anja",
    "Rafael", "Fatima", "Jonas", "Elif", "Owen", "Beatriz", "Mateo",
    "Sinead", "Kenji", "Zara", "Anders", "Camille", "Noah", "Rukmini", "Sven",
)
LAST_NAMES: tuple[str, ...] = (
    "Alvarez", "Novak", "Sorensen", "Adeyemi", "Fischer", "Rossi", "Haddad",
    "Nakamura", "Kowalski", "Dubois", "Brennan", "Petrov", "Silva", "Mensah",
    "Lindqvist", "Vargas", "Okafor", "Reyes", "Kaur", "Bauer", "Marchetti",
    "Halloran", "Osei", "Ferreira", "Iqbal", "Jansen", "Moreau", "Tanaka",
    "Weber", "Donnelly",
)
CITIES: tuple[str, ...] = (
    "Bristol", "Porto", "Tallinn", "Nairobi", "Osaka", "Rotterdam", "Calgary",
    "Valencia", "Gdansk", "Adelaide", "Montpellier", "Cork", "Bergen",
    "Ljubljana", "Hamilton",
)
COMPANIES: tuple[str, ...] = (
    "Fenwick Logistics", "Blue Harrow Studio", "Ostara Foods", "Pinewell Clinic",
    "Larkspur Interiors", "Trelane Engineering", "Harborlight Group",
    "Quillon Analytics", "Verdant Rail", "Castermill Works",
)
ROLES: tuple[str, ...] = (
    "backend engineer", "support specialist", "data analyst",
    "product designer", "site reliability engineer",
)
OUTLETS: tuple[str, ...] = (
    "the Weekly Ledger", "Northgate Review", "Circuit Weekly",
    "the Meridian Post", "Foundry Quarterly",
)
MONTHS: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
)


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """One department/urgency pair, with wording variants for each sentence.

    A ticket is one subject, one request and one detail drawn from these, plus a
    shared urgency sentence chosen by `severity`. The department and the urgency
    of the rendered ticket are the template's, which is what makes ground truth
    exact rather than labelled.
    """

    id: str
    department: str
    severity: str
    subjects: tuple[str, ...]
    requests: tuple[str, ...]
    details: tuple[str, ...]


TEMPLATES: tuple[Template, ...] = (
    # ---- billing ---------------------------------------------------------
    Template(
        id="billing_address_update",
        department="billing",
        severity="low",
        subjects=(
            "Billing address change for future invoices",
            "Updating the billing details on our invoices",
            "Small change to our invoice details",
        ),
        requests=(
            "Could you update the billing address we hold so that future invoices go to our new office?",
            "Please change the billing address on our invoices to our new office.",
            "I would like the billing address printed on our invoices changed to the new office.",
        ),
        details=(
            "The old one is still shown on invoice {invoice}, and we would like our cost centre code added while you are in there.",
            "Invoice {invoice} still shows the previous office, and we would like our cost centre code shown as well.",
            "Everything else on invoice {invoice} is correct; only the address and the cost centre code need changing.",
        ),
    ),
    Template(
        id="billing_invoice_copy",
        department="billing",
        severity="normal",
        subjects=(
            "Copy of invoice {invoice}",
            "Need invoice {invoice} again for expenses",
            "Duplicate invoice request",
        ),
        requests=(
            "Could you send me another copy of invoice {invoice}? I need it for an expenses claim.",
            "Please resend invoice {invoice}; our finance team needs it before they will process my expenses.",
            "I cannot find invoice {invoice} anywhere and need another copy for an expenses claim.",
        ),
        details=(
            "The payment itself went through on {date} for {amount}, so this is only about the paperwork.",
            "The card payment of {amount} cleared on {date}; I only need the document itself.",
            "It was charged to the card on {date} for {amount}, so nothing is outstanding.",
        ),
    ),
    Template(
        id="billing_card_declined",
        department="billing",
        severity="high",
        subjects=(
            "Payment declined and the subscription lapses today",
            "Card payment failing on invoice {invoice}",
            "Our card was declined and the subscription is at risk",
        ),
        requests=(
            "Our card payment was declined this morning and the subscription lapses today, so please retry the charge on the replacement card.",
            "The subscription payment failed and I need the charge taken on a different card before the subscription lapses.",
            "Payment on the subscription was declined; please switch to the replacement card and retry the charge.",
        ),
        details=(
            "The bank has lifted the block on the replacement card, and the invoice is {invoice} for {amount}.",
            "The replacement card works everywhere else; the outstanding amount is {amount} on invoice {invoice}.",
            "The failed amount is {amount} against invoice {invoice}, and the new card is already on file.",
        ),
    ),
    Template(
        id="billing_double_charge",
        department="billing",
        severity="critical",
        subjects=(
            "Charged three times for invoice {invoice}",
            "Duplicate charges still being taken",
            "We have been charged {amount} three times",
        ),
        requests=(
            "We have been charged {amount} three times for invoice {invoice} and the payments are still going out, so please stop them and refund the duplicates.",
            "Three identical charges of {amount} have come off the card against invoice {invoice}; please reverse the duplicates.",
            "The card has been charged {amount} three times for invoice {invoice} and a fourth payment attempt is pending.",
        ),
        details=(
            "Our bank shows the latest payment as pending, so money is still leaving as I write this.",
            "Two of the charges have cleared and a third is pending, which has taken us past our overdraft limit.",
            "The duplicated payments have left us short for payroll this week.",
        ),
    ),
    # ---- shipping --------------------------------------------------------
    Template(
        id="shipping_delivery_preference",
        department="shipping",
        severity="low",
        subjects=(
            "Delivery preference for future parcels",
            "Please leave future parcels at reception",
            "Change to our delivery instructions",
        ),
        requests=(
            "Could you add a note to our delivery instructions so the courier leaves future parcels at reception rather than the loading bay?",
            "Please update the delivery instructions so parcels are handed in at reception instead of left at the loading bay.",
            "For future deliveries, could the courier be asked to bring parcels to reception?",
        ),
        details=(
            "The parcel from {order} arrived fine, so nothing has gone wrong; this is only about where things are left.",
            "{order} was delivered without any problem; I am just trying to save the courier a walk next time.",
            "Nothing is missing from {order}; it is only the drop-off point I would like changed.",
        ),
    ),
    Template(
        id="shipping_tracking_stalled",
        department="shipping",
        severity="normal",
        subjects=(
            "Tracking has not moved since {date}",
            "No tracking update on {order}",
            "Shipment appears stuck in transit",
        ),
        requests=(
            "The tracking for {order} has not updated since {date} and I would like to know where the shipment actually is.",
            "Could you chase the courier about {order}? The tracking has shown the same depot since {date}.",
            "Tracking on {order} has been stuck at the same depot since {date}; please find out what has happened to the parcel.",
        ),
        details=(
            "The courier's page shows it left the {city} depot and nothing since.",
            "It was scanned out of the {city} depot and has not been scanned anywhere else.",
            "The last scan was at the {city} depot and there has been no movement at all.",
        ),
    ),
    Template(
        id="shipping_wrong_address",
        department="shipping",
        severity="high",
        subjects=(
            "Parcel going to our old address today",
            "Redirect needed on {order}",
            "Delivery is heading to the wrong address",
        ),
        requests=(
            "{order} is out for delivery to our old address, so please have the courier redirect it before it is dropped off.",
            "The parcel for {order} is on the van for the wrong address; please redirect the delivery before the courier arrives.",
            "Please ask the courier to redirect the parcel for {order}; it is being delivered to an address we left months ago.",
        ),
        details=(
            "The tracking says it is fifteen stops away, so there is still time to catch it.",
            "Tracking shows the van a few streets from the old address.",
            "The courier's page says delivery is expected within the hour.",
        ),
    ),
    Template(
        id="shipping_event_stock_missing",
        department="shipping",
        severity="critical",
        subjects=(
            "Shipment for tomorrow's event is not at the venue",
            "Missing delivery for an event tomorrow morning",
            "{order} has not arrived and the event opens tomorrow",
        ),
        requests=(
            "The shipment for {order} was due at the venue yesterday and is not here; the event opens tomorrow morning and we need the parcels found.",
            "{order} has not been delivered to the venue and we open tomorrow morning, so please trace the shipment now.",
            "We are missing the entire delivery for {order} at the venue and the event starts tomorrow.",
        ),
        details=(
            "The courier's line says the pallet is sitting at the {city} depot with no driver assigned.",
            "The courier told us the parcels were routed to the {city} depot by mistake and nobody is moving them.",
            "Tracking shows the shipment back at the {city} depot after a failed delivery attempt.",
        ),
    ),
    # ---- technical -------------------------------------------------------
    Template(
        id="technical_chart_label_overlap",
        department="technical",
        severity="low",
        subjects=(
            "Small display bug on the reports screen",
            "Chart labels overlap in the dashboard",
            "Cosmetic bug in the dashboard charts",
        ),
        requests=(
            "There is a small bug on the reports screen: the chart labels overlap each other once the window is narrow.",
            "I wanted to report a cosmetic bug -- the labels on the dashboard charts sit on top of one another at narrow widths.",
            "The dashboard charts have a display bug where the axis labels overlap when the browser window is made narrow.",
        ),
        details=(
            "No error appears and nothing is broken; the numbers themselves are right.",
            "There is no error message and the underlying figures are correct.",
            "It throws no error at all and the data is fine; it only looks untidy.",
        ),
    ),
    Template(
        id="technical_csv_export_bug",
        department="technical",
        severity="normal",
        subjects=(
            "CSV export drops the last row",
            "CSV export is missing a row",
            "Bug in the CSV export",
        ),
        requests=(
            "The CSV export is dropping the last row of every report, which looks like an off-by-one bug.",
            "Every CSV export we run is missing its final row, and that looks like a bug rather than our data.",
            "There is a bug in the CSV export: the last row never makes it into the file.",
        ),
        details=(
            "I checked the same report on screen and the row is there, so it happens on the way out.",
            "The row shows on screen and in the api response, so something drops it when the file is written.",
            "The api returns the row, so the problem is in the file itself.",
        ),
    ),
    Template(
        id="technical_sync_failure",
        department="technical",
        severity="high",
        subjects=(
            "Desktop client will not sync -- error {ref}",
            "Sync failing all morning with an error",
            "Sync error is stopping me working",
        ),
        requests=(
            "The desktop client has failed to sync all morning and shows error {ref} every time.",
            "Sync on the desktop client fails immediately with error {ref}, so none of my work is going anywhere.",
            "Every sync attempt on the desktop client ends in error {ref} and nothing uploads.",
        ),
        details=(
            "I have restarted it, cleared the local cache and tried another machine; the same error comes back.",
            "A restart and a fresh install both end at the same error, so it is not this machine.",
            "The log file shows the same timeout each time before the error appears.",
        ),
    ),
    Template(
        id="technical_api_outage",
        department="technical",
        severity="critical",
        subjects=(
            "api returning 503 for every request",
            "Production integration is down with api errors",
            "All api calls failing since this morning",
        ),
        requests=(
            "Every call to the api endpoint has returned a 503 error for the last twenty minutes and our production integration is down.",
            "The api is returning 503 on every endpoint and our live integration has stopped working entirely.",
            "All of our api requests are hitting a timeout or a 503 error, and production traffic is failing.",
        ),
        details=(
            "Our monitoring shows the errors started at 09:40 and the error rate is still 100 per cent.",
            "The error rate went from zero to 100 per cent at 09:40 and has stayed there.",
            "Our retries fail too; the stack trace is the same timeout every time.",
        ),
    ),
    # ---- account access --------------------------------------------------
    Template(
        id="account_recovery_email",
        department="account_access",
        severity="low",
        subjects=(
            "Changing the recovery email on my login",
            "Please update my recovery email",
            "Recovery email is out of date",
        ),
        requests=(
            "Could you change the recovery email on my login to {email}? The old one no longer exists.",
            "Please update the recovery email attached to my login to {email}.",
            "I would like the recovery email for my login changed to {email}.",
        ),
        details=(
            "I can still sign in with my password and two-factor code perfectly well.",
            "My password works fine, so nothing is broken at the moment.",
            "My password and authenticator both still work; I only want the recovery email corrected.",
        ),
    ),
    Template(
        id="account_2fa_new_phone",
        department="account_access",
        severity="normal",
        subjects=(
            "Moving my authenticator to a new phone",
            "Two-factor re-enrolment needed",
            "New phone, need two-factor set up again",
        ),
        requests=(
            "I have a new phone and need my two-factor authenticator enrolled on it again.",
            "Could you clear my two-factor enrolment so I can set the authenticator up on my new phone?",
            "My authenticator codes are stuck on my old phone; please re-enrol two-factor on the new one.",
        ),
        details=(
            "I am still signed in on my desktop, so I am not locked out while we sort this.",
            "My desktop session is still active, so nothing is blocked in the meantime.",
            "I still have a working session on my laptop and can wait for this to be done properly.",
        ),
    ),
    Template(
        id="account_locked_out",
        department="account_access",
        severity="high",
        subjects=(
            "Locked out after too many attempts",
            "Cannot log in -- locked out",
            "Login locked and I need access today",
        ),
        requests=(
            "I am locked out after too many failed sign-in attempts and cannot log in at all.",
            "My login is locked after several failed attempts and the password reset link does nothing.",
            "Too many wrong password attempts have locked me out and I cannot get back in.",
        ),
        details=(
            "The unlock link in your email takes me back to the same locked screen.",
            "Changing the password does not clear the lock; I end up at the same screen.",
            "I have waited the thirty minutes it suggests and the lock is still there.",
        ),
    ),
    Template(
        id="account_takeover",
        department="account_access",
        severity="critical",
        subjects=(
            "Unrecognised login from another country",
            "Someone else is signed in as me",
            "Possible unauthorised access to my login",
        ),
        requests=(
            "There is an active session from a country I have never been to, so please revoke every session and force a password reset.",
            "Someone I do not recognise is signed in as me right now; revoke all sessions and reset my credentials.",
            "I can see an unfamiliar active session on my login and I need every session revoked and the password changed.",
        ),
        details=(
            "The unfamiliar session appeared at 02:10 and is still listed as active.",
            "Two-factor prompts have been arriving on their own for the last hour.",
            "I have changed the password twice and the other session is still there.",
        ),
    ),
    # ---- sales -----------------------------------------------------------
    Template(
        id="sales_roadmap_question",
        department="sales",
        severity="low",
        subjects=(
            "Evaluating your product -- one question before we buy",
            "Pre-purchase question from {company}",
            "Roadmap question while we evaluate",
        ),
        requests=(
            "We are evaluating your product for next year and, before we commit to buying anything, I would like to know whether a Spanish interface is planned.",
            "Before we go ahead and buy, could you tell me whether a Spanish interface is on the roadmap? We are still evaluating.",
            "We are still evaluating vendors and would like to know whether a Spanish interface is planned before we choose one.",
        ),
        details=(
            "We are not a customer yet; this is for a decision we will take in the autumn.",
            "Nothing has been bought yet and the decision will not be made until the autumn.",
            "We have no contract with you at the moment; this is groundwork for an autumn decision.",
        ),
    ),
    Template(
        id="sales_quote_request",
        department="sales",
        severity="normal",
        subjects=(
            "Quote for {seats} seats",
            "Written quote needed for procurement",
            "Pricing for {seats} seats at {company}",
        ),
        requests=(
            "Could you send a written quote for {seats} seats? Our procurement team needs it on paper before they will raise anything.",
            "We need a formal quote for {seats} seats so that procurement can raise the purchase order.",
            "Please send pricing for {seats} seats as a written quote; procurement will not move without one.",
        ),
        details=(
            "We have never bought from you before, so there is nothing on record at your end.",
            "This would be our first contract with you, starting in the new quarter.",
            "We would be a new customer and the budget is approved for the new quarter.",
        ),
    ),
    Template(
        id="sales_order_form_deadline",
        department="sales",
        severity="high",
        subjects=(
            "Order form needed today for procurement",
            "Signed order form needed before end of day",
            "Purchase order stuck without your paperwork",
        ),
        requests=(
            "Procurement needs your countersigned order form today or the budget for this goes back into the pot.",
            "We cannot raise the purchase order without your signed order form, and procurement closes the budget today.",
            "Our procurement team needs the countersigned order form from you today to release the purchase order.",
        ),
        details=(
            "The quote you sent for {seats} seats is agreed; it is only the paperwork that is missing.",
            "Everything in the quote for {seats} seats is agreed internally and only your signature is outstanding.",
            "The {seats} seat quote is approved on our side and nothing else is holding it up.",
        ),
    ),
    Template(
        id="sales_contract_window_closing",
        department="sales",
        severity="critical",
        subjects=(
            "Vendor portal closes in three hours",
            "Contract lapses this afternoon without your signature",
            "Purchase order window closing today",
        ),
        requests=(
            "The vendor portal holding our purchase order closes in three hours, and without your signed contract uploaded by then the whole thing falls through for this financial year.",
            "Our vendor portal shuts in three hours; without your countersigned contract uploaded, the purchase order is cancelled for the year.",
            "If the contract is not signed and uploaded to the vendor portal within three hours, procurement cancels the purchase order entirely.",
        ),
        details=(
            "I have the {seats} seat quote and everything else ready to attach.",
            "Everything else for the {seats} seat contract is ready; only your signature is missing.",
            "Our side is signed, so the {seats} seat deal fails on this one document.",
        ),
    ),
    # ---- privacy ---------------------------------------------------------
    Template(
        id="privacy_retention_question",
        department="privacy",
        severity="low",
        subjects=(
            "How long do you keep support transcripts?",
            "Retention question about personal data",
            "Question about your retention policy",
        ),
        requests=(
            "Could you tell me how long you keep the personal data in support conversations? I am writing our own retention policy and want to line it up with yours.",
            "I am drafting a retention policy and would like to know your retention period for the personal data held in support conversations.",
            "For our own records, what is your retention period for the personal data in support conversations?",
        ),
        details=(
            "This is not a formal data request; I am only after the policy.",
            "I am not asking you to do anything with my data, only to describe the policy.",
            "Nothing needs to be deleted or produced; the policy itself is all I need.",
        ),
    ),
    Template(
        id="privacy_access_request",
        department="privacy",
        severity="normal",
        subjects=(
            "Data subject access request",
            "Request for a copy of my personal data",
            "Formal data request under data protection law",
        ),
        requests=(
            "I am making a data subject access request for a copy of all the personal data you hold about me.",
            "Please treat this as a formal data request: I would like a copy of all the personal data you hold about me.",
            "Under data protection law I am asking for a copy of all personal data you hold on me.",
        ),
        details=(
            "Please confirm you have received this so the statutory period is on record.",
            "A note back to say it has arrived would be helpful so that both of us have the date written down.",
            "I understand you have a month to answer; I would just like to know it has arrived.",
        ),
    ),
    Template(
        id="privacy_erasure_deadline",
        department="privacy",
        severity="high",
        subjects=(
            "Erasure request deadline is tomorrow morning",
            "Outstanding erasure request, deadline tomorrow",
            "Data protection deadline on my erasure request",
        ),
        requests=(
            "I made an erasure request a month ago and have heard nothing; the statutory deadline falls tomorrow morning.",
            "My erasure request from a month ago is still outstanding and the data protection deadline is tomorrow morning.",
            "The erasure request I made under data protection law is nearly out of time; the deadline is tomorrow morning.",
        ),
        details=(
            "I would rather not take this to the regulator, but I will have to if the deadline passes.",
            "If the deadline passes I will have no choice but to raise it with the regulator.",
            "I still have the acknowledgement you sent on {date} with the deadline printed on it.",
        ),
    ),
    Template(
        id="privacy_misdirected_data",
        department="privacy",
        severity="critical",
        subjects=(
            "I have been sent another customer's personal data",
            "Someone else's personal data has arrived in my inbox",
            "Data protection incident: wrong attachment",
        ),
        requests=(
            "The message you sent me this morning has another customer's personal data attached to it, and the file is sitting in my inbox now.",
            "Your last email to me contained a file full of another customer's personal data, which looks like a data protection incident.",
            "I have just been sent someone else's personal data by mistake and need to know what to do with it.",
        ),
        details=(
            "It lists names, home addresses and phone numbers for about forty people.",
            "The file has names and home addresses for dozens of people who are not me.",
            "There are names, addresses and phone numbers for around forty individuals in it.",
        ),
    ),
    # ---- careers ---------------------------------------------------------
    Template(
        id="careers_future_openings",
        department="careers",
        severity="low",
        subjects=(
            "Keeping my cv on file for future openings",
            "No suitable job right now, but for the future",
            "Interested in future positions",
        ),
        requests=(
            "There is no position open that fits me at the moment, so could you keep my cv on file for future openings?",
            "I could not find a job on your careers page that fits, and wondered whether you keep a cv on file for future positions.",
            "Nothing on your job listings suits me right now; may I send my cv in for future openings anyway?",
        ),
        details=(
            "I have eight years behind me as a {role} and would be glad to hear about anything similar.",
            "My background is eight years as a {role}.",
            "I have spent the last eight years working as a {role}.",
        ),
    ),
    Template(
        id="careers_application_status",
        department="careers",
        severity="normal",
        subjects=(
            "Status of my application for the {role} position",
            "Following up on my job application",
            "Any update on my application?",
        ),
        requests=(
            "I applied for the {role} position three weeks ago and have heard nothing; could someone in recruiting tell me where my application stands?",
            "Three weeks after applying for the {role} job I still have no news, so I wanted to ask whether my application is still being considered.",
            "Could recruiting let me know whether my application for the {role} position is still live? I applied three weeks ago.",
        ),
        details=(
            "The reference on it is {ref}, submitted on {date}.",
            "I applied on {date} and the reference given was {ref}.",
            "The reference I was given is {ref} and the date was {date}.",
        ),
    ),
    Template(
        id="careers_reschedule_interview",
        department="careers",
        severity="high",
        subjects=(
            "Need to move tomorrow's interview",
            "Interview tomorrow -- can we reschedule?",
            "Rescheduling my interview",
        ),
        requests=(
            "My interview is tomorrow morning and I have to move it, so could the recruiter confirm a new time today?",
            "I need to reschedule tomorrow's interview and need confirmation from the recruiter today.",
            "Tomorrow's interview clashes with something I cannot move; can recruiting rebook it and confirm today?",
        ),
        details=(
            "Any slot later in the week works for me; the reference is {ref}.",
            "I can do any time later in the week. My reference is {ref}.",
            "Later in the week is wide open for me and my reference is {ref}.",
        ),
    ),
    Template(
        id="careers_competing_offer",
        department="careers",
        severity="critical",
        subjects=(
            "Competing offer expires tonight",
            "Need an answer from recruiting today",
            "Decision deadline tonight on my application",
        ),
        requests=(
            "I have a written offer from another company that expires at midnight tonight, and I need recruiting to tell me whether one is coming from you.",
            "Another employer's offer lapses tonight, so I need recruiting to say today whether I am still a candidate.",
            "My other offer expires tonight and I would rather join you, so can the recruiter tell me where my application stands before then?",
        ),
        details=(
            "My final interview was on {date} and the reference is {ref}.",
            "I finished the last interview on {date}; the reference is {ref}.",
            "The interview rounds finished on {date} and I have heard nothing since.",
        ),
    ),
    # ---- press -----------------------------------------------------------
    Template(
        id="press_media_kit",
        department="press",
        severity="low",
        subjects=(
            "Media kit request for a roundup",
            "Press assets for a piece next month",
            "Logo and press kit request",
        ),
        requests=(
            "I am a journalist putting together a roundup for {outlet} next month and would like your press kit and logo files.",
            "Could you send the press kit over? I am writing a roundup for {outlet} that runs next month.",
            "For a roundup in {outlet} next month I need your media kit and a high-resolution logo.",
        ),
        details=(
            "Nothing is on deadline; the piece is not scheduled until the new month.",
            "There is no deadline pressure here, since publication is weeks away.",
            "The piece will not be published for several weeks yet.",
        ),
    ),
    Template(
        id="press_background_briefing",
        department="press",
        severity="normal",
        subjects=(
            "Background briefing for a feature in {outlet}",
            "Request for a background conversation",
            "Press request: feature in {outlet}",
        ),
        requests=(
            "I am a reporter at {outlet} working on a feature and would like a background briefing, with something on the record afterwards.",
            "Could I arrange a background briefing for a feature I am writing for {outlet}?",
            "I write for {outlet} and would like to set up a background briefing with someone from your team.",
        ),
        details=(
            "The piece is scheduled for the end of the month, so there is time to arrange it properly.",
            "Publication is at the end of the month, so any slot in the next fortnight works.",
            "My editor has it down for the end of the month.",
        ),
    ),
    Template(
        id="press_fact_check",
        department="press",
        severity="high",
        subjects=(
            "Fact-check before the piece files today",
            "Checking a figure before we publish",
            "Press fact-check, filing today",
        ),
        requests=(
            "I am fact-checking a figure for a piece in {outlet} that files to my editor at the end of the day and I need it confirmed before then.",
            "Before I file to my editor tonight, could you confirm one figure for the piece running in {outlet}?",
            "My editor takes the copy at the end of today and I need one number confirmed on the record before I file.",
        ),
        details=(
            "The figure I have is {amount} and I would rather print it right than print it fast.",
            "I am about to print {amount} and would rather check it with you than guess.",
            "The number in question is {amount}, from a source I would like you to confirm.",
        ),
    ),
    Template(
        id="press_statement_deadline",
        department="press",
        severity="critical",
        subjects=(
            "Publishing in two hours, statement needed",
            "Press request before publication this afternoon",
            "Story runs at 17:00 and needs your comment",
        ),
        requests=(
            "We are publishing a story about your company in two hours and I need a statement from you before it goes out.",
            "The story runs in two hours and I have to carry either your statement or a line saying you declined.",
            "My editor publishes in two hours and I need your comment on the record before then.",
        ),
        details=(
            "I have sent the relevant lines to your general inbox twice already today.",
            "I tried your newsroom line twice this morning with no answer.",
            "The piece names your company, so I would rather have your words in it than a note that you did not reply.",
        ),
    ),
)

TEMPLATES_BY_ID: dict[str, Template] = {t.id: t for t in TEMPLATES}
TEMPLATES_BY_DEPARTMENT: dict[str, tuple[Template, ...]] = {
    d: tuple(t for t in TEMPLATES if t.department == d) for d in DEPARTMENTS
}


# --------------------------------------------------------------------------
# Ticket rendering
# --------------------------------------------------------------------------

_BASE_PADDING: dict[str, int] = {"clean": 0, "noisy": 2, "hard": 3}


@dataclass(frozen=True)
class Ticket:
    """One rendered ticket, with the label it was rendered from.

    E3, E4 and E8 use this type directly rather than going through `generate()`,
    because they need the ticket as raw material -- as a labelled exemplar, as a
    state to dilute, as filler around a target question -- rather than as a
    finished instance.
    """

    template_id: str
    department: str
    severity: str
    level: str
    subject: str | None
    body: str
    customer: str
    customer_email: str
    received: str
    padding_sentences: int
    distractor_departments: tuple[str, ...]

    @property
    def text(self) -> str:
        """Subject and body as one string, for keyword counting and for E4's
        terse-versus-verbose comparison."""
        return f"{self.subject}\n\n{self.body}" if self.subject is not None else self.body

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def as_state(self) -> dict:
        """The `state` payload: an object, since the docs recommend structured
        state and E4 compares it against its own stringified form.

        Carries nothing that names the department or the urgency -- no template
        id, no label, no queue name -- so the state cannot leak the answer.
        """
        state: dict[str, Any] = {
            "channel": "email",
            "received": self.received,
            "from": {"name": self.customer, "email": self.customer_email},
        }
        if self.subject is not None:
            state["subject"] = self.subject
        state["body"] = self.body
        return state


def _ascii_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _fields(rng: random.Random) -> dict[str, str]:
    """Surface detail for one ticket.

    Every field is drawn on every ticket whether the template uses it or not, so
    that the number of draws -- and therefore the rest of the ticket -- does not
    depend on which template was selected.
    """
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    company = rng.choice(COMPANIES)
    return {
        "name": f"{first} {last}",
        "first_name": first,
        "email": f"{first.lower()}.{_ascii_slug(last)}@{_ascii_slug(company.split()[0])}.example",
        "company": company,
        "city": rng.choice(CITIES),
        "order": f"ORD-{rng.randrange(100000, 1000000)}",
        "invoice": f"INV-{rng.randrange(10000, 100000)}",
        "ref": "REF-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5)),
        "own_ref": "CUS-" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(5)),
        "amount": f"${rng.randrange(1800, 480000) / 100:,.2f}",
        "date": f"{rng.randrange(1, 29)} {rng.choice(MONTHS)}",
        "seats": str(rng.randrange(8, 120)),
        "role": rng.choice(ROLES),
        "outlet": rng.choice(OUTLETS),
        "teammate": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
        "year": str(rng.randrange(2017, 2025)),
        "received": f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
    }


def _typo(word: str, rng: random.Random) -> str:
    """Transpose two adjacent interior letters.

    Interior only, and one word per sentence at most: a transposition in the
    middle of a word costs a reader nothing, which is what keeps the noisy and
    hard levels difficult for a string matcher and easy for a person.

    Letters only, and the two must differ. Swapping a letter with the hyphen or
    the trailing punctuation next to it turns "same-day" into "sam-eday" and
    "drop-off" into "dro-poff", which is not a typo a reader forgives at a
    glance; swapping a letter with its twin leaves the word alone and quietly
    spends the sentence's one typo on nothing.
    """
    positions = [
        i
        for i in range(1, len(word) - 2)
        if word[i].isalpha() and word[i + 1].isalpha() and word[i] != word[i + 1]
    ]
    if not positions:
        return word
    i = rng.choice(positions)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _roughen(sentence: str, rng: random.Random) -> str:
    """The surface noise for the noisy and hard levels: one typo and sometimes a
    lost capital. Nothing that changes meaning.

    Sentence-final punctuation is left alone here and dropped in `_paragraphs`
    instead, where it can be dropped only at the end of a paragraph -- a missing
    full stop in the middle of one runs two sentences together, and a reader
    should never have to work out where a sentence ended.
    """
    words = sentence.split(" ")
    # Reference codes and amounts are left alone: a transposition inside
    # "INV-64613" looks like a different invoice rather than like a typo.
    candidates = [
        i
        for i, w in enumerate(words)
        if len(w.strip(".,;:?-")) >= 6 and not any(c.isdigit() or c == "$" for c in w)
    ]
    if candidates and rng.random() < 0.45:
        i = rng.choice(candidates)
        words[i] = _typo(words[i], rng)
    out = " ".join(words)
    if rng.random() < 0.30:
        out = out[0].lower() + out[1:]
    return out


def _neutral_sentences(rng: random.Random, n: int, fields: dict[str, str]) -> list[str]:
    """`n` padding sentences, drawn without replacement until the pool runs out."""
    out: list[str] = []
    while len(out) < n:
        pool = list(NEUTRAL_SENTENCES)
        rng.shuffle(pool)
        out.extend(pool[: n - len(out)])
    return [s.format(**fields) for s in out]


def _subject(template: Template, rng: random.Random, level: str, fields: dict[str, str]) -> str | None:
    if level == "hard":
        # The subject never carries the routing signal at this level, so the
        # answer has to come out of the body.
        if rng.random() < 0.4:
            return None
        return rng.choice(VAGUE_SUBJECTS)
    subject = rng.choice(template.subjects).format(**fields)
    if level == "clean":
        return subject
    roll = rng.random()
    if roll < 0.15:
        return None
    if roll < 0.60:
        return subject.lower()
    return subject


def _paragraphs(sentences: list[str], rng: random.Random, *, drop_final_stop: bool = False) -> str:
    out: list[str] = []
    i = 0
    while i < len(sentences):
        k = rng.randint(2, 4)
        paragraph = " ".join(sentences[i : i + k])
        if drop_final_stop and paragraph.endswith(".") and rng.random() < 0.3:
            paragraph = paragraph[:-1]
        out.append(paragraph)
        i += k
    return "\n\n".join(out)


def render_ticket(
    template: Template,
    rng: random.Random,
    *,
    level: str = "clean",
    padding_sentences: int = 0,
    padding_position: str = "tail",
) -> Ticket:
    """Render one ticket from `template`.

    `padding_sentences` is the length knob E4 sweeps: neutral sentences that say
    nothing about which department should handle the ticket and nothing about how
    urgent it is, so growing a ticket from 80 words to several thousand leaves
    both answers untouched. `padding_position` decides whether they go before the
    request, after it, or both.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    if padding_position not in PADDING_POSITIONS:
        raise ValueError(f"unknown padding_position {padding_position!r}")
    if padding_sentences < 0:
        raise ValueError("padding_sentences must be >= 0")

    f = _fields(rng)
    request = rng.choice(template.requests).format(**f)
    detail = rng.choice(template.details).format(**f)
    urgency = rng.choice(URGENCY_SENTENCES[template.severity])

    base_pad = _BASE_PADDING[level]
    pads = _neutral_sentences(rng, base_pad + padding_sentences, f)
    base_pads, extra_pads = pads[:base_pad], pads[base_pad:]

    distractor_departments: tuple[str, ...] = ()
    distractors: list[str] = []
    if level == "hard":
        others = [d for d in DEPARTMENTS if d != template.department]
        rng.shuffle(others)
        distractor_departments = tuple(others[:2])
        distractors = [rng.choice(DISTRACTOR_SENTENCES[d]) for d in distractor_departments]

    if level == "clean":
        body_sentences = [request, detail, urgency]
    elif level == "noisy":
        body_sentences = [request, base_pads[0], detail, urgency, base_pads[1]]
    else:
        # The request sits in the middle, behind padding and two explicit
        # disclaimers about other departments.
        body_sentences = [
            base_pads[0],
            distractors[0],
            base_pads[1],
            request,
            detail,
            distractors[1],
            base_pads[2],
            urgency,
        ]

    if extra_pads:
        if padding_position == "head":
            body_sentences = extra_pads + body_sentences
        elif padding_position == "tail":
            body_sentences = body_sentences + extra_pads
        else:
            half = len(extra_pads) // 2
            body_sentences = extra_pads[:half] + body_sentences + extra_pads[half:]

    if level != "clean":
        body_sentences = [_roughen(s, rng) for s in body_sentences]

    subject = _subject(template, rng, level, f)
    greeting = rng.choice(GREETINGS)
    closer = rng.choice(CLOSERS)
    paragraphs = _paragraphs(body_sentences, rng, drop_final_stop=level != "clean")
    body = "\n\n".join([greeting, paragraphs, f"{closer}\n{f['name']}"])

    return Ticket(
        template_id=template.id,
        department=template.department,
        severity=template.severity,
        level=level,
        subject=subject,
        body=body,
        customer=f["name"],
        customer_email=f["email"],
        received=f["received"],
        padding_sentences=padding_sentences,
        distractor_departments=distractor_departments,
    )


# --------------------------------------------------------------------------
# Assignment: which department and which template instance `i` gets
# --------------------------------------------------------------------------


def assignment_for(*, seed: int, index: int) -> tuple[str, Template]:
    """The (department, template) for instance `index`.

    Balanced by construction rather than by sampling: each block of 8 consecutive
    indices contains each department exactly once, in a seeded order, and each
    department's four templates -- one per urgency level -- cycle over successive
    blocks. So any prefix whose length is a multiple of 8 is exactly balanced
    across departments, any multiple of 32 is exactly balanced across urgency
    levels as well, and the majority-class baseline is the random baseline. Both
    depend on `index` alone, so instance 400 is still reproducible on its own.

    Deliberately independent of level and padding, so the same index is the same
    ticket at every rung of the ladder and E4 can grow a state without also
    changing the underlying ticket.
    """
    block, pos = divmod(index, len(DEPARTMENTS))
    order = list(DEPARTMENTS)
    rng_for(NAME, {"assign": "department", "block": block}, seed).shuffle(order)
    department = order[pos]

    templates = list(TEMPLATES_BY_DEPARTMENT[department])
    sub_block, sub_pos = divmod(block, len(templates))
    rng_for(NAME, {"assign": "template", "department": department, "block": sub_block}, seed).shuffle(templates)
    return department, templates[sub_pos]


def ticket_for(
    *,
    seed: int,
    index: int,
    level: str = "clean",
    padding_sentences: int = 0,
    padding_position: str = "tail",
) -> Ticket:
    """The ticket for one instance coordinate, with no question attached.

    This is the entry point for E3, E4 and E8. The ticket depends on the level
    and the padding but not on the question type, so the same index gives the
    same state whether it is being asked a noul, a choice or a score.
    """
    _, template = assignment_for(seed=seed, index=index)
    rng = rng_for(
        NAME,
        {"level": level, "padding_sentences": padding_sentences, "padding_position": padding_position},
        seed,
        index,
    )
    return render_ticket(
        template,
        rng,
        level=level,
        padding_sentences=padding_sentences,
        padding_position=padding_position,
    )


def tickets(
    *,
    seed: int,
    count: int,
    level: str = "clean",
    padding_sentences: int = 0,
    padding_position: str = "tail",
) -> list[Ticket]:
    """`count` tickets, for callers that want labelled text rather than
    instances -- E8's in-context exemplar blocks, E3's filler material."""
    return [
        ticket_for(
            seed=seed,
            index=i,
            level=level,
            padding_sentences=padding_sentences,
            padding_position=padding_position,
        )
        for i in range(count)
    ]


# --------------------------------------------------------------------------
# Questions and instances
# --------------------------------------------------------------------------

# Keys are neutral. E1 renames them to check the docs' claim that they are not
# used in inference, and that test is only meaningful if the original key does
# not hint at the answer.
KEY_NOUL = "q_route_check"
KEY_CHOICE = "q_route"
KEY_SCORE = "q_urgency"

NOUL_QUESTION = "Should this support ticket be handled by the {label} team?"
CHOICE_QUESTION = "Which team should handle this support ticket?"
SCORE_QUESTION = "How urgent is this support ticket?"


def _normalize(difficulty: dict) -> dict:
    """Canonical difficulty dict.

    Normalized rather than taken as given because `difficulty` feeds both
    `Instance.instance_id` and the per-instance RNG, so the same condition
    written two ways has to produce the same instances. The padding keys are
    dropped when there is no padding, which keeps the common condition's id and
    its label in the report short.
    """
    d = dict(difficulty)
    level = d.pop("level", DEFAULT_DIFFICULTY["level"])
    question_type = d.pop("question_type", DEFAULT_DIFFICULTY["question_type"])
    padding_sentences = int(d.pop("padding_sentences", 0))
    padding_position = d.pop("padding_position", "tail")
    if d:
        raise ValueError(f"unknown difficulty keys: {sorted(d)}")
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {LEVELS}")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unknown question_type {question_type!r}; expected one of {QUESTION_TYPES}")
    if padding_position not in PADDING_POSITIONS:
        raise ValueError(f"unknown padding_position {padding_position!r}")
    if padding_sentences < 0:
        raise ValueError("padding_sentences must be >= 0")
    out = {"level": level, "question_type": question_type}
    if padding_sentences:
        out["padding_sentences"] = padding_sentences
        out["padding_position"] = padding_position
    return out


def _questions(
    ticket: Ticket, question_type: str, index: int, rng: random.Random
) -> tuple[dict[str, dict], dict[str, Any], dict[str, Any]]:
    questions: dict[str, dict] = {}
    truth: dict[str, Any] = {}
    extra_meta: dict[str, Any] = {}

    if question_type in ("noul", "all"):
        # Polarity from the index rather than the RNG, so any prefix of a
        # condition is exactly balanced yes/no and the baseline is exactly 0.5.
        answer = index % 2 == 0
        if answer:
            asked = ticket.department
        else:
            asked = rng.choice([d for d in DEPARTMENTS if d != ticket.department])
        questions[KEY_NOUL] = noul(NOUL_QUESTION.format(label=DEPARTMENT_LABELS[asked]))
        truth[KEY_NOUL] = answer
        extra_meta["asked_department"] = asked

    if question_type in ("choice", "all"):
        options = [{"id": d, "label": DEPARTMENT_LABELS[d]} for d in DEPARTMENTS]
        questions[KEY_CHOICE] = shuffled_options(choice(CHOICE_QUESTION, options), rng)
        truth[KEY_CHOICE] = ticket.department

    if question_type in ("score", "all"):
        # The rubric is ordered, so it is not shuffled.
        questions[KEY_SCORE] = score(SCORE_QUESTION, SEVERITY_RUBRIC)
        truth[KEY_SCORE] = ticket.severity

    return questions, truth, extra_meta


def _instance(difficulty: dict, seed: int, index: int) -> Instance:
    level = difficulty["level"]
    padding_sentences = difficulty.get("padding_sentences", 0)
    padding_position = difficulty.get("padding_position", "tail")
    ticket = ticket_for(
        seed=seed,
        index=index,
        level=level,
        padding_sentences=padding_sentences,
        padding_position=padding_position,
    )
    rng = rng_for(NAME, difficulty, seed, index)
    questions, truth, extra_meta = _questions(ticket, difficulty["question_type"], index, rng)

    counts = keyword_counts(ticket.text)
    meta: dict[str, Any] = {
        "level": level,
        "question_type": difficulty["question_type"],
        "template_id": ticket.template_id,
        "department": ticket.department,
        "severity": ticket.severity,
        "word_count": ticket.word_count,
        "padding_sentences": padding_sentences,
        "distractor_departments": list(ticket.distractor_departments),
        "has_subject": ticket.subject is not None,
        # What the baselines need. `metrics.majority_baseline` takes the labels
        # and `metrics.random_baseline` takes the class count, so neither rate is
        # stored here; what cannot be recovered from the log is the cheap
        # heuristic's view of the ticket, so that is what goes in.
        "n_departments": len(DEPARTMENTS),
        "n_rubric_points": len(SEVERITIES),
        "baseline_keyword_counts": counts,
        "baseline_keyword_prediction": keyword_baseline(counts),
    }
    meta.update(extra_meta)

    instance = Instance(
        generator=NAME,
        # A copy per instance: the dict reaches the JSONL and an experiment that
        # annotates one instance's difficulty should not edit the whole condition.
        difficulty=dict(difficulty),
        seed=seed,
        index=index,
        state=ticket.as_state(),
        questions=questions,
        truth=truth,
        meta=meta,
    )
    instance.validate()
    return instance


def generate(*, difficulty: dict, seed: int, count: int) -> list[Instance]:
    """`count` instances at one difficulty setting.

    `difficulty` takes "level" (clean, noisy, hard), "question_type" (noul,
    choice, score, or "all" for one state carrying all three), and optionally
    "padding_sentences" with "padding_position" for the length sweep.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    d = _normalize(difficulty)
    return [_instance(d, seed, i) for i in range(count)]


_PADDING_KEYS = frozenset({"padding_sentences", "padding_position"})


def _typed(question_type: str, level: str, padding: dict) -> dict:
    """Difficulty for one of the per-type wrappers.

    `**padding` is spread into the difficulty dict, so without this check
    `generate_choice(..., question_type="score")` would quietly return score
    instances from a function named for choice -- a whole condition logged under
    the wrong question type. The wrappers exist to fix the type, so anything that
    would move it is an error rather than an override.
    """
    unknown = sorted(set(padding) - _PADDING_KEYS)
    if unknown:
        raise ValueError(
            f"generate_{question_type}() takes only {sorted(_PADDING_KEYS)} "
            f"besides seed, count and level; got {unknown}"
        )
    return {"level": level, "question_type": question_type, **padding}


def generate_noul(
    *, seed: int, count: int, level: str = "clean", **padding: Any
) -> list[Instance]:
    """Balanced yes/no: does this ticket belong to the named department."""
    return generate(difficulty=_typed("noul", level, padding), seed=seed, count=count)


def generate_choice(
    *, seed: int, count: int, level: str = "clean", **padding: Any
) -> list[Instance]:
    """Pick one of the 8 departments. Option order is shuffled per instance."""
    return generate(difficulty=_typed("choice", level, padding), seed=seed, count=count)


def generate_score(
    *, seed: int, count: int, level: str = "clean", **padding: Any
) -> list[Instance]:
    """Place the ticket on the 4-point urgency rubric."""
    return generate(difficulty=_typed("score", level, padding), seed=seed, count=count)


def difficulty_sweep() -> list[dict]:
    """The ladder, easy to hard.

    Three levels of wording by three question types. The levels are the real
    axis; the question types are there because the control has to demonstrate
    that all three round-trip, and because a failure confined to one question
    type is a different diagnosis from a failure across all three. Within a
    level they are ordered by baseline, 0.5 then 0.25 then 0.125.

    The "all" question type is not in the sweep: it exists for the Phase 0 smoke
    test, which wants all three types against one state in a handful of calls.
    """
    return [
        {"level": level, "question_type": qt}
        for level in LEVELS
        for qt in ("noul", "score", "choice")
    ]


# --------------------------------------------------------------------------
# Self-check
# --------------------------------------------------------------------------
#
# The control is only worth anything if it really is unambiguous, so the claim
# is checked mechanically rather than asserted in a comment: no template sentence
# carries another department's vocabulary, no shared sentence carries any, and
# every distractor names exactly the one department it rules out.


def _hits(text: str) -> dict[str, int]:
    return {d: n for d, n in keyword_counts(text).items() if n}


def _check_vocabulary() -> None:
    for t in TEMPLATES:
        for sentence in t.subjects + t.requests + t.details:
            foreign = {d: n for d, n in _hits(sentence).items() if d != t.department}
            assert not foreign, f"{t.id}: foreign vocabulary {foreign} in {sentence!r}"
        for request in t.requests:
            for detail in t.details:
                counts = keyword_counts(f"{request} {detail}")
                assert counts[t.department] > 0, f"{t.id}: no own vocabulary in {request!r} {detail!r}"

    neutral = list(NEUTRAL_SENTENCES) + list(GREETINGS) + list(CLOSERS) + list(VAGUE_SUBJECTS)
    for pool in URGENCY_SENTENCES.values():
        neutral.extend(pool)
    for sentence in neutral:
        assert not _hits(sentence), f"shared sentence carries {_hits(sentence)}: {sentence!r}"

    for department, pool in DISTRACTOR_SENTENCES.items():
        for sentence in pool:
            assert set(_hits(sentence)) == {department}, (
                f"distractor for {department} carries {_hits(sentence)}: {sentence!r}"
            )

    # Exhaustively over the pools rather than only over draws from them: a
    # sample cannot prove that the one company name carrying "invoice" is absent.
    for pool_name, pool in (
        ("FIRST_NAMES", FIRST_NAMES), ("LAST_NAMES", LAST_NAMES), ("CITIES", CITIES),
        ("COMPANIES", COMPANIES), ("ROLES", ROLES), ("OUTLETS", OUTLETS),
        ("MONTHS", MONTHS),
    ):
        for value in pool:
            assert not _hits(value), f"{pool_name} entry {value!r} carries {_hits(value)}"
    # And over composed values -- the email address, the reference codes and the
    # amounts, which no pool lists.
    rng = random.Random(0)
    for _ in range(200):
        for key, value in _fields(rng).items():
            assert not _hits(value), f"field {key}={value!r} carries {_hits(value)}"


def _check_templates() -> None:
    assert len(DEPARTMENTS) == 8, "the docstring's 0.125 baseline assumes 8 departments"
    assert len(TEMPLATES_BY_ID) == len(TEMPLATES), "duplicate template id"
    for department in DEPARTMENTS:
        templates = TEMPLATES_BY_DEPARTMENT[department]
        assert len(templates) == len(SEVERITIES), f"{department}: expected one template per severity"
        assert sorted(t.severity for t in templates) == sorted(SEVERITIES), (
            f"{department}: severities are {[t.severity for t in templates]}"
        )
    for t in TEMPLATES:
        assert t.department in DEPARTMENT_LABELS, t.id
        assert min(len(t.subjects), len(t.requests), len(t.details)) >= 3, t.id


# Everything above checks the pools the ticket is built from. What follows
# checks the text a model actually sees, by recovering the answer from that text
# instead of trusting the label the renderer carried through -- "the department
# is whatever the template said" is not a check of anything.
#
# Sorting a word's letters makes `_typo`'s transposition and a lost capital
# invisible, so a template sentence can be found in a roughened ticket by exact
# set containment rather than by a fuzzy match with a threshold to argue about.

_CONTENT_WORD = re.compile(r"[A-Za-z]{4,}")


def _fingerprint(text: str) -> set[str]:
    return {"".join(sorted(w.lower())) for w in _CONTENT_WORD.findall(text)}


def _sentence_words(sentence: str) -> set[str]:
    """`_fingerprint` of a template sentence, with its {field} slots removed."""
    return _fingerprint(re.sub(r"\{[a-z_]+\}", " ", sentence))


def _shared_words() -> set[str]:
    """Every word that can reach a ticket from somewhere other than a template.

    Subtracting these from a template's sentences leaves the wording only that
    template could have produced, which is what makes the recovery unambiguous.
    """
    out: set[str] = set()
    pools: list[tuple[str, ...]] = [NEUTRAL_SENTENCES, GREETINGS, CLOSERS, VAGUE_SUBJECTS]
    pools.extend(URGENCY_SENTENCES.values())
    pools.extend(DISTRACTOR_SENTENCES.values())
    for pool in pools:
        for sentence in pool:
            out |= _sentence_words(sentence)
    return out


def _template_variants(template: Template, shared: set[str]) -> list[set[str]]:
    """The template-only wording of each (request, detail) pair."""
    return [
        (_sentence_words(request) | _sentence_words(detail)) - shared
        for request in template.requests
        for detail in template.details
    ]


def _check_rendered(seed: int) -> None:
    """Recover the answer from the rendered ticket and compare it to the truth.

    Three things have to hold of the text itself: exactly one template could
    have produced it, exactly one urgency sentence is in it, and the vocabulary
    the cheap baseline reads is confined to the department that owns the ticket
    plus, at the hard level, the two departments the ticket explicitly rules out.
    """
    shared = _shared_words()
    variants = {t.id: _template_variants(t, shared) for t in TEMPLATES}
    for template_id, pairs in variants.items():
        # An empty set is a subset of every ticket, so a pair with no wording of
        # its own would match everything and make the recovery meaningless.
        assert all(pairs), f"{template_id}: a (request, detail) pair has no wording of its own"

    lost = 0
    total = 0
    for level in LEVELS:
        for padding in (0, 9):
            difficulty: dict[str, Any] = {"level": level, "question_type": "all"}
            if padding:
                difficulty["padding_sentences"] = padding
                difficulty["padding_position"] = "split"
            for instance in generate(difficulty=difficulty, seed=seed, count=64):
                total += 1
                where = f"{level}/pad{padding}/{instance.index}"
                state = instance.state
                text = f"{state.get('subject', '')}\n{state['body']}"
                seen = _fingerprint(text)

                found = [tid for tid, pairs in variants.items()
                         if any(words <= seen for words in pairs)]
                assert found == [instance.meta["template_id"]], (
                    f"{where}: the text matches {found}, labelled {instance.meta['template_id']}"
                )
                department = TEMPLATES_BY_ID[found[0]].department
                assert instance.truth[KEY_CHOICE] == department, where
                assert instance.truth[KEY_NOUL] == (
                    instance.meta["asked_department"] == department
                ), where

                severities = [
                    severity
                    for severity, pool in URGENCY_SENTENCES.items()
                    if any(_sentence_words(s) <= seen for s in pool)
                ]
                assert severities == [instance.truth[KEY_SCORE]], (
                    f"{where}: the text carries urgency {severities}, "
                    f"labelled {instance.truth[KEY_SCORE]}"
                )

                named = list(instance.meta["distractor_departments"])
                expected = 2 if level == "hard" else 0
                assert len(named) == expected, (
                    f"{where}: meta names {named} as distractors at level {level}"
                )
                assert department not in named, f"{where}: ruled out its own department"

                hits = set(_hits(text))
                allowed = {department} | set(named)
                assert hits <= allowed, f"{where}: vocabulary {sorted(hits - allowed)} from nowhere"
                if level == "clean":
                    assert hits == {department}, f"{where}: clean ticket carries {sorted(hits)}"
                if department not in hits:
                    lost += 1

    # A typo can land on the one keyword carrying the department, leaving the
    # cheap baseline nothing to read. A reader still routes the ticket, since the
    # request is still there in words, but the rate has to stay small or the
    # keyword column measures the noise rather than the level. Measured at about
    # 0.2% at noisy and 0.9% at hard over eight seeds.
    assert lost <= 0.05 * total, f"{lost} of {total} tickets lost their own vocabulary"


# The bracket the ladder exists to produce, asserted rather than only printed:
# the control is worth something because of the gap between what comprehension
# gets and what string matching gets, so a template edit that closes the gap has
# to fail here rather than quietly print a smaller number. Clean is exact -- no
# typos and no distractors, so the only vocabulary in the ticket is its own --
# and the rest are the ranges seen over forty seeds at this sample size, widened.
_BASELINE_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("clean", "noul"): (1.0, 1.0),
    ("clean", "choice"): (1.0, 1.0),
    ("noisy", "noul"): (0.95, 1.0),
    ("noisy", "choice"): (0.95, 1.0),
    ("hard", "noul"): (0.50, 0.85),
    ("hard", "choice"): (0.25, 0.65),
}


def _check_generation(seed: int) -> list[tuple]:
    import json
    from collections import Counter

    count = 64  # two full cycles of 32: exact balance over departments and severities
    rows: list[tuple] = []

    for difficulty in difficulty_sweep():
        question_type = difficulty["question_type"]
        instances = generate(difficulty=difficulty, seed=seed, count=count)
        assert len(instances) == count

        departments: Counter = Counter()
        severities: Counter = Counter()
        yes_answers = 0
        baseline_correct = 0
        words = 0

        for instance in instances:
            meta = instance.meta
            template = TEMPLATES_BY_ID[meta["template_id"]]
            assert template.department == meta["department"]
            assert template.severity == meta["severity"]
            departments[meta["department"]] += 1
            severities[meta["severity"]] += 1
            words += meta["word_count"]

            # The state must not carry the answer in any form.
            blob = json.dumps(instance.state)
            assert meta["template_id"] not in blob
            assert "department" not in blob.lower()

            prediction = meta["baseline_keyword_prediction"]
            if question_type == "choice":
                assert instance.truth[KEY_CHOICE] == template.department
                option_ids = sorted(o["id"] for o in instance.questions[KEY_CHOICE]["options"])
                assert option_ids == sorted(DEPARTMENTS)
                baseline_correct += prediction == template.department
            elif question_type == "noul":
                asked = meta["asked_department"]
                assert instance.truth[KEY_NOUL] == (asked == template.department)
                assert DEPARTMENT_LABELS[asked] in instance.questions[KEY_NOUL]["question"]
                yes_answers += instance.truth[KEY_NOUL]
                baseline_correct += (prediction == asked) == instance.truth[KEY_NOUL]
            else:
                assert instance.truth[KEY_SCORE] == template.severity
                assert [r["id"] for r in instance.questions[KEY_SCORE]["rubric"]] == SEVERITIES

        assert set(departments) == set(DEPARTMENTS), departments
        assert set(departments.values()) == {count // len(DEPARTMENTS)}, departments
        assert set(severities.values()) == {count // len(SEVERITIES)}, severities
        if question_type == "noul":
            assert yes_answers == count // 2, yes_answers

        baseline = None if question_type == "score" else baseline_correct / count
        if baseline is not None:
            low, high = _BASELINE_BOUNDS[(difficulty["level"], question_type)]
            assert low <= baseline <= high, (
                f"{difficulty['level']}/{question_type}: keyword baseline {baseline:.3f} "
                f"is outside [{low}, {high}] -- the ladder has moved"
            )
        rows.append(
            (difficulty["level"], question_type, count, words / count, baseline)
        )
    return rows


def _check_determinism(seed: int) -> None:
    import json

    def blob(instances: list[Instance]) -> str:
        return json.dumps([i.to_json() for i in instances], sort_keys=True)

    for difficulty in difficulty_sweep() + [{"level": "hard", "question_type": "all"}]:
        first = generate(difficulty=difficulty, seed=seed, count=24)
        second = generate(difficulty=difficulty, seed=seed, count=24)
        assert blob(first) == blob(second), f"not deterministic at {difficulty}"
        # Instance i depends on i alone, not on how many were drawn before it.
        assert blob(generate(difficulty=difficulty, seed=seed, count=5)) == blob(first[:5])
        assert len({i.instance_id for i in first}) == len(first)

    # The three question types must sit on identical states, or the smoke test
    # is comparing three different tasks.
    for level in LEVELS:
        states = {
            qt: [i.state for i in generate(difficulty={"level": level, "question_type": qt}, seed=seed, count=8)]
            for qt in QUESTION_TYPES
        }
        for qt, value in states.items():
            assert value == states["noul"], f"{level}/{qt}: state differs from the noul condition"


def _check_padding(seed: int) -> None:
    base = ticket_for(seed=seed, index=3, level="clean")
    for position in PADDING_POSITIONS:
        padded = ticket_for(
            seed=seed, index=3, level="clean", padding_sentences=40, padding_position=position
        )
        assert padded.department == base.department and padded.severity == base.severity
        assert padded.template_id == base.template_id
        assert padded.word_count > 3 * base.word_count
        # Padding is neutral, so it moves neither answer nor the cheap baseline.
        assert keyword_baseline(keyword_counts(padded.text)) == base.department


def _self_check() -> None:
    seed = 4242
    _check_templates()
    _check_vocabulary()
    _check_determinism(seed)
    _check_padding(seed)
    _check_rendered(seed)
    rows = _check_generation(seed)

    print(f"{NAME}: {len(TEMPLATES)} templates, {len(DEPARTMENTS)} departments, "
          f"{len(SEVERITIES)} urgency levels")
    print(f"baselines: choice {RANDOM_BASELINE_CHOICE:.3f}  noul {RANDOM_BASELINE_NOUL:.3f}  "
          f"score {RANDOM_BASELINE_SCORE:.3f}  (balanced by construction, so majority == random)")
    print()
    print(f"{'level':<7} {'question':<9} {'n':>4} {'mean words':>11} {'keyword baseline':>17}")
    for level, question_type, count, mean_words, baseline in rows:
        cell = "--" if baseline is None else f"{baseline:.3f}"
        print(f"{level:<7} {question_type:<9} {count:>4} {mean_words:>11.0f} {cell:>17}")
    print()
    example = ticket_for(seed=seed, index=11, level="hard")
    print(f"example ({example.department}/{example.severity}, level hard):")
    print(example.text)


if __name__ == "__main__":
    _self_check()
