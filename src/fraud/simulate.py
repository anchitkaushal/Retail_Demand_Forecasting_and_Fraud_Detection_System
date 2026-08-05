# src/fraud/simulate.py

# src/fraud/simulate.py

from pathlib import Path

import numpy as np
import pandas as pd


def simulate_fraud_data(
    n_transactions=10000,
    fraud_rate=0.05,
    random_state=42,
    output_path="data/simulated/fraud_transactions.csv"
):
    """
    Generate a synthetic retail return/fraud dataset.

    Fraudulent transactions intentionally overlap with
    normal transactions so that the classification problem
    is not artificially easy.
    """

    rng = np.random.default_rng(
        random_state
    )

    # ------------------------------------------------
    # Number of transactions
    # ------------------------------------------------

    n_fraud = int(
        n_transactions * fraud_rate
    )

    n_normal = (
        n_transactions -
        n_fraud
    )

    # =================================================
    # NORMAL TRANSACTIONS
    # =================================================

    normal = pd.DataFrame({

        "transaction_id": range(
            n_normal
        ),

        # Normal customers generally purchase
        # relatively small quantities, but some
        # larger purchases are possible.
        "quantity": np.clip(
            rng.poisson(
                2.5,
                n_normal
            ) + 1,
            1,
            12
        ),

        # Transaction amounts have substantial overlap
        # with fraudulent transactions.
        "transaction_amount": np.clip(
            rng.lognormal(
                mean=3.8,
                sigma=0.65,
                size=n_normal
            ),
            5,
            500
        ),

        # Most returns happen within a reasonable period,
        # but some normal returns can also happen quickly.
        "return_days": np.clip(
            rng.normal(
                loc=12,
                scale=7,
                size=n_normal
            ),
            0,
            30
        ),

        # Normal customers can occasionally have
        # relatively high purchase frequency.
        "customer_frequency": np.clip(
            rng.poisson(
                4,
                n_normal
            ) + 1,
            1,
            25
        ),

        # Most customers have few previous returns,
        # but some can have several.
        "previous_returns": np.clip(
            rng.poisson(
                1.5,
                n_normal
            ),
            0,
            10
        ),

        "is_fraud": 0
    })

    # =================================================
    # FRAUDULENT TRANSACTIONS
    # =================================================

    fraud = pd.DataFrame({

        "transaction_id": range(
            n_normal,
            n_transactions
        ),

        # Fraudulent transactions tend to have somewhat
        # higher quantities, but overlap with normal ones.
        "quantity": np.clip(
            rng.poisson(
                4,
                n_fraud
            ) + 1,
            1,
            15
        ),

        # Fraud transactions tend to be larger, but there
        # is intentional overlap with normal transactions.
        "transaction_amount": np.clip(
            rng.lognormal(
                mean=4.4,
                sigma=0.75,
                size=n_fraud
            ),
            10,
            800
        ),

        # Fraudulent returns are somewhat more likely
        # to happen quickly.
        "return_days": np.clip(
            rng.normal(
                loc=6,
                scale=6,
                size=n_fraud
            ),
            0,
            30
        ),

        # Fraudulent customers tend to transact more
        # frequently, but normal customers can overlap.
        "customer_frequency": np.clip(
            rng.poisson(
                7,
                n_fraud
            ) + 1,
            1,
            30
        ),

        # Fraudulent customers tend to have more previous
        # returns, but the distributions overlap.
        "previous_returns": np.clip(
            rng.poisson(
                3,
                n_fraud
            ),
            0,
            15
        ),

        "is_fraud": 1
    })

    # =================================================
    # Combine
    # =================================================

    df = pd.concat(
        [
            normal,
            fraud
        ],
        ignore_index=True
    )

    # Shuffle transactions
    df = df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(
        drop=True
    )

    # =================================================
    # Save
    # =================================================

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Fraud dataset saved to {output_path}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["is_fraud"].value_counts()
    )

    print(
        "\nClass percentages:"
    )

    print(
        df["is_fraud"]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .round(2)
    )

    return df