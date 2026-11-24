import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VISUALS_DIR = BASE_DIR / "visuals"

DATA_DIR.mkdir(exist_ok=True)
VISUALS_DIR.mkdir(exist_ok=True)


print("Generating dataset...")

N_USERS = 5000

events = [
    "app_open",
    "registration",
    "view_product",
    "add_to_cart",
    "purchase"
]

sources = [
    "google",
    "instagram",
    "facebook",
    "tiktok",
    "organic"
]

users = []

for user_id in range(1, N_USERS + 1):

    registration_date = (
        pd.Timestamp("2025-01-01")
        + pd.Timedelta(
            days=np.random.randint(0, 180)
        )
    )

    source = np.random.choice(
        sources,
        p=[0.25, 0.20, 0.15, 0.15, 0.25]
    )

    experiment_group = np.random.choice(
        ["control", "test"]
    )

    users.append({
        "user_id": user_id,
        "registration_date": registration_date,
        "source": source,
        "experiment_group": experiment_group
    })

users_df = pd.DataFrame(users)


event_rows = []

for _, user in users_df.iterrows():

    user_id = user["user_id"]
    reg_date = user["registration_date"]

    event_rows.append({
        "user_id": user_id,
        "event_date": reg_date,
        "event": "app_open"
    })

    if np.random.random() < 0.90:

        registration_date = (
            reg_date
            + pd.Timedelta(
                minutes=np.random.randint(1, 1440)
            )
        )

        event_rows.append({
            "user_id": user_id,
            "event_date": registration_date,
            "event": "registration"
        })

        if np.random.random() < 0.75:

            product_date = (
                registration_date
                + pd.Timedelta(
                    hours=np.random.randint(1, 48)
                )
            )

            event_rows.append({
                "user_id": user_id,
                "event_date": product_date,
                "event": "view_product"
            })

            if np.random.random() < 0.55:

                cart_date = (
                    product_date
                    + pd.Timedelta(
                        hours=np.random.randint(1, 24)
                    )
                )

                event_rows.append({
                    "user_id": user_id,
                    "event_date": cart_date,
                    "event": "add_to_cart"
                })

                if np.random.random() < 0.45:

                    purchase_date = (
                        cart_date
                        + pd.Timedelta(
                            hours=np.random.randint(1, 48)
                        )
                    )

                    event_rows.append({
                        "user_id": user_id,
                        "event_date": purchase_date,
                        "event": "purchase"
                    })


events_df = pd.DataFrame(event_rows)

events_df["event_date"] = pd.to_datetime(
    events_df["event_date"]
)

events_df.to_csv(
    DATA_DIR / "app_events.csv",
    index=False
)

users_df.to_csv(
    DATA_DIR / "users.csv",
    index=False
)

print(f"Users: {len(users_df)}")
print(f"Events: {len(events_df)}")


print("\n================ DATA QUALITY ================")

print("\nMissing values:")

print(events_df.isna().sum())

print("\nDuplicate rows:")

print(events_df.duplicated().sum())

events_df = events_df.drop_duplicates()


total_users = users_df["user_id"].nunique()

registered_users = events_df.loc[
    events_df["event"] == "registration",
    "user_id"
].nunique()

product_view_users = events_df.loc[
    events_df["event"] == "view_product",
    "user_id"
].nunique()

cart_users = events_df.loc[
    events_df["event"] == "add_to_cart",
    "user_id"
].nunique()

purchase_users = events_df.loc[
    events_df["event"] == "purchase",
    "user_id"
].nunique()


registration_rate = (
    registered_users / total_users * 100
)

product_view_rate = (
    product_view_users / registered_users * 100
)

cart_rate = (
    cart_users / product_view_users * 100
)

purchase_rate = (
    purchase_users / cart_users * 100
)


print("\n================ PRODUCT KPIs ================")

print(f"Users: {total_users}")
print(f"Registered: {registered_users}")
print(f"Product viewers: {product_view_users}")
print(f"Added to cart: {cart_users}")
print(f"Purchased: {purchase_users}")

print(
    f"\nRegistration conversion: "
    f"{registration_rate:.2f}%"
)

print(
    f"Product view conversion: "
    f"{product_view_rate:.2f}%"
)

print(
    f"Cart conversion: "
    f"{cart_rate:.2f}%"
)

print(
    f"Purchase conversion from cart: "
    f"{purchase_rate:.2f}%"
)


funnel = pd.DataFrame({
    "stage": [
        "App Open",
        "Registration",
        "Product View",
        "Add to Cart",
        "Purchase"
    ],
    "users": [
        total_users,
        registered_users,
        product_view_users,
        cart_users,
        purchase_users
    ]
})

print("\n================ FUNNEL ================")

print(funnel)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=funnel,
    x="stage",
    y="users"
)

plt.title("User Conversion Funnel")

plt.xlabel("Stage")

plt.ylabel("Users")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    VISUALS_DIR / "funnel.png"
)

plt.close()

source_analysis = []

for source in sources:

    source_users = users_df[
        users_df["source"] == source
    ]["user_id"]

    source_purchases = events_df[
        (events_df["user_id"].isin(source_users)) &
        (events_df["event"] == "purchase")
    ]["user_id"].nunique()

    source_total = len(source_users)

    conversion = (
        source_purchases / source_total * 100
    )

    source_analysis.append({
        "source": source,
        "users": source_total,
        "purchases": source_purchases,
        "conversion": conversion
    })


source_df = pd.DataFrame(source_analysis)

source_df = source_df.sort_values(
    "conversion",
    ascending=False
)

print("\n================ TRAFFIC SOURCES ================")

print(source_df)


plt.figure(figsize=(10, 6))

sns.barplot(
    data=source_df,
    x="source",
    y="conversion"
)

plt.title("Purchase Conversion by Traffic Source")

plt.ylabel("Conversion (%)")

plt.xlabel("Source")

plt.tight_layout()

plt.savefig(
    VISUALS_DIR / "source_conversion.png"
)

plt.close()


events_df["date"] = (
    events_df["event_date"].dt.date
)

daily_active_users = (
    events_df
    .groupby("date")["user_id"]
    .nunique()
    .reset_index(name="DAU")
)

print("\n================ DAU ================")

print(
    daily_active_users.head(10)
)


plt.figure(figsize=(12, 6))

sns.lineplot(
    data=daily_active_users,
    x="date",
    y="DAU"
)

plt.title("Daily Active Users")

plt.xlabel("Date")

plt.ylabel("DAU")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    VISUALS_DIR / "dau.png"
)

plt.close()


events_df["event_date"] = pd.to_datetime(
    events_df["event_date"]
)

first_events = (
    events_df
    .groupby("user_id")["event_date"]
    .min()
    .reset_index(name="first_date")
)

events_df = events_df.merge(
    first_events,
    on="user_id"
)

events_df["days_since_first"] = (
    events_df["event_date"]
    - events_df["first_date"]
).dt.days


retention_days = [1, 7, 14, 30]

retention_results = []

for day in retention_days:

    users_on_day = events_df[
        events_df["days_since_first"] == day
    ]["user_id"].nunique()

    retention = (
        users_on_day / total_users * 100
    )

    retention_results.append({
        "day": day,
        "retention": retention
    })


retention_df = pd.DataFrame(
    retention_results
)

print("\n================ RETENTION ================")

print(retention_df)


plt.figure(figsize=(8, 5))

sns.lineplot(
    data=retention_df,
    x="day",
    y="retention",
    marker="o"
)

plt.title("User Retention")

plt.xlabel("Days After First Activity")

plt.ylabel("Retention (%)")

plt.tight_layout()

plt.savefig(
    VISUALS_DIR / "retention.png"
)

plt.close()

experiment = []

for group in ["control", "test"]:

    group_users = users_df[
        users_df["experiment_group"] == group
    ]["user_id"]

    purchases = events_df[
        (events_df["user_id"].isin(group_users)) &
        (events_df["event"] == "purchase")
    ]["user_id"].nunique()

    total = len(group_users)

    conversion = (
        purchases / total * 100
    )

    experiment.append({
        "group": group,
        "users": total,
        "purchases": purchases,
        "conversion": conversion
    })


ab_df = pd.DataFrame(experiment)

print("\n================ A/B TEST ================")

print(ab_df)

control_conversion = (
    ab_df.loc[
        ab_df["group"] == "control",
        "conversion"
    ].iloc[0]
)

test_conversion = (
    ab_df.loc[
        ab_df["group"] == "test",
        "conversion"
    ].iloc[0]
)

uplift = (
    (test_conversion - control_conversion)
    / control_conversion * 100
)

print(
    f"\nConversion uplift: {uplift:.2f}%"
)


plt.figure(figsize=(7, 5))

sns.barplot(
    data=ab_df,
    x="group",
    y="conversion"
)

plt.title("A/B Test: Purchase Conversion")

plt.xlabel("Experiment Group")

plt.ylabel("Conversion (%)")

plt.tight_layout()

plt.savefig(
    VISUALS_DIR / "ab_test.png"
)

plt.close()


db_path = BASE_DIR / "app_analytics.db"

connection = sqlite3.connect(
    db_path
)

users_df.to_sql(
    "users",
    connection,
    if_exists="replace",
    index=False
)

events_df.to_sql(
    "events",
    connection,
    if_exists="replace",
    index=False
)


query = """

SELECT
    u.source,

    COUNT(DISTINCT u.user_id) AS users,

    COUNT(
        DISTINCT CASE
            WHEN e.event = 'purchase'
            THEN e.user_id
        END
    ) AS buyers,

    ROUND(
        COUNT(
            DISTINCT CASE
                WHEN e.event = 'purchase'
                THEN e.user_id
            END
        ) * 100.0
        / COUNT(DISTINCT u.user_id),
        2
    ) AS conversion_rate

FROM users u

LEFT JOIN events e
    ON u.user_id = e.user_id

GROUP BY u.source

ORDER BY conversion_rate DESC;

"""


sql_result = pd.read_sql_query(
    query,
    connection
)

print("\n================ SQL ANALYSIS ================")

print(
    sql_result.to_string(index=False)
)



best_source = source_df.iloc[0]["source"]

best_source_conversion = (
    source_df.iloc[0]["conversion"]
)

print("\n================ BUSINESS INSIGHTS ================")

print(
    f"1. Best traffic source: {best_source}"
)

print(
    f"   Conversion: "
    f"{best_source_conversion:.2f}%"
)

print(
    f"\n2. Overall purchase conversion: "
    f"{purchase_users / total_users * 100:.2f}%"
)

print(
    f"\n3. A/B test uplift: "
    f"{uplift:.2f}%"
)

print(
    "\n4. Main analytical task:"
)

print(
    "Identify the funnel stage with the largest "
    "drop-off and test improvements there."
)

print(
    "\n5. Marketing recommendation:"
)

print(
    f"Prioritize analysis of the {best_source} "
    "traffic channel because it has the highest "
    "purchase conversion."
)


connection.close()

print("\n======================================")
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("======================================")