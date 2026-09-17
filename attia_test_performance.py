'''
commandline:
#python your_script.py \
--testing-results ~/path/attia_kfold_testing_results.xlsx \
--demograph ~/path/ptb/ptb_scp_diag_simplified.csv \
--output ~/path/attia_testing_demographic_analysis.xlsx
'''

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# default arguments
DEFAULT_TESTING_RESULTS_PATH = "~/path/attia_kfold_testing_results.xlsx"
DEFAULT_DEMOGRAPH_PATH = "~/path/ptb/ptb_scp_diag_simplified.csv"
DEFAULT_OUTPUT_DIR = "~/path/attia_testing_demographic_analysis.xlsx"

WEIGHT_SOURCE_UNIT = "lb"
WEIGHT_BIN_SIZE_LB = 20.0

SEX_LABELS = {
    0: "Male",
    1: "Female",
}

FOLD_SHEET_NAMES = [f"Fold_{i}" for i in range(5)]


def save_figure(fig, title, output_dir):
    """Generate a filename based on the Figure Title, save it as a .jpg image, and store it in the same parent directory as `output_dir`."""
    parent_dir = Path(output_dir).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    
    # Replace ':' and ' ' in the title with '_' and add the .jpg suffix.
    safe_title = title.replace(":", "_").replace(" ", "_")
    file_path = parent_dir / f"{safe_title}.jpg"
    
    fig.savefig(file_path, format="jpg", dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {file_path}")


def load_input_data(testing_results_path, demograph_path):
    testing_results = {
        sheet: pd.read_excel(
            testing_results_path,
            sheet_name=sheet,
        )
        for sheet in FOLD_SHEET_NAMES
    }

    demograph = pd.read_csv(demograph_path)

    required_demo = [
        "ecg_id",
        "patient_id",
        "age",
        "sex",
        "weight",
    ]

    required_test = [
        "ecg_id",
        "true_label",
        "probability",
        "threshold",
        "prediction",
    ]

    missing_demo = [
        col for col in required_demo
        if col not in demograph.columns
    ]
    if missing_demo:
        raise ValueError(
            "Missing demographic columns: "
            + ", ".join(missing_demo)
        )

    for sheet, frame in testing_results.items():
        missing_test = [
            col for col in required_test
            if col not in frame.columns
        ]
        if missing_test:
            raise ValueError(
                f"{sheet} is missing columns: "
                + ", ".join(missing_test)
            )

    return testing_results, demograph


def calculate_testing_metrics(frame):
    y_true = frame["true_label"].to_numpy(dtype=int)
    y_prob = frame["probability"].to_numpy(dtype=float)
    y_pred = frame["prediction"].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn > 0 else np.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else np.nan
    auc = (
        roc_auc_score(y_true, y_prob)
        if len(np.unique(y_true)) == 2
        else np.nan
    )

    return {
        "n_test": int(len(frame)),
        "n_error": int(np.sum(y_true != y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auc": float(auc),
        "threshold": float(frame["threshold"].iloc[0]),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def build_testing_summary(testing_results):
    rows = []
    for fold_name, frame in testing_results.items():
        metrics = calculate_testing_metrics(frame)
        rows.append({"fold": fold_name, **metrics})
    return pd.DataFrame(rows)


def make_age_groups(age):
    age_numeric = pd.to_numeric(age, errors="coerce")
    valid = age_numeric.dropna()

    if valid.empty:
        return pd.Series([np.nan] * len(age), index=age.index)

    minimum = max(0, int(np.floor(valid.min() / 10.0) * 10))
    maximum = int(np.floor(valid.max() / 10.0) * 10)

    edges = np.arange(minimum, maximum + 21, 10, dtype=float)
    labels = [f"{int(x)}-{int(x + 9)}" for x in edges[:-1]]

    return pd.cut(
        age_numeric,
        bins=edges,
        right=False,
        labels=labels,
        include_lowest=True,
    )


def make_weight_groups(weight):
    weight_numeric = pd.to_numeric(weight, errors="coerce")

    if WEIGHT_SOURCE_UNIT.lower() == "kg":
        weight_lb = weight_numeric * 2.20462262185
    elif WEIGHT_SOURCE_UNIT.lower() == "lb":
        weight_lb = weight_numeric
    else:
        raise ValueError(
            "WEIGHT_SOURCE_UNIT must be 'kg' or 'lb'."
        )

    valid = weight_lb.dropna()
    if valid.empty:
        return pd.Series([np.nan] * len(weight), index=weight.index)

    minimum = (
        np.floor(valid.min() / WEIGHT_BIN_SIZE_LB)
        * WEIGHT_BIN_SIZE_LB
    )
    maximum = (
        np.floor(valid.max() / WEIGHT_BIN_SIZE_LB)
        * WEIGHT_BIN_SIZE_LB
    )

    edges = np.arange(
        minimum,
        maximum + WEIGHT_BIN_SIZE_LB * 2,
        WEIGHT_BIN_SIZE_LB,
    )

    labels = [
        f"{int(x)}-{int(x + WEIGHT_BIN_SIZE_LB - 1)} lb"
        for x in edges[:-1]
    ]

    return pd.cut(
        weight_lb,
        bins=edges,
        right=False,
        labels=labels,
        include_lowest=True,
    )


def add_demographics(testing_frame, demograph):
    demo_columns = [
        "ecg_id",
        "patient_id",
        "age",
        "sex",
        "weight",
    ]

    merged = testing_frame.merge(
        demograph[demo_columns],
        on="ecg_id",
        how="left",
        validate="one_to_one",
    )

    merged["error"] = (
        merged["true_label"] != merged["prediction"]
    ).astype(int)

    merged["age_group"] = make_age_groups(merged["age"])
    merged["weight_group"] = make_weight_groups(merged["weight"])
    merged["sex_label"] = (
        merged["sex"]
        .map(SEX_LABELS)
        .fillna("Unknown")
    )

    return merged


def build_demographic_overall(demograph):
    age = pd.to_numeric(demograph["age"], errors="coerce")
    weight = pd.to_numeric(demograph["weight"], errors="coerce")

    weight_lb = (
        weight * 2.20462262185
        if WEIGHT_SOURCE_UNIT.lower() == "kg"
        else weight
    )

    rows = [
        ["Age", "N", age.notna().sum()],
        ["Age", "Mean", age.mean()],
        ["Age", "Std", age.std()],
        ["Age", "Min", age.min()],
        ["Age", "Median", age.median()],
        ["Age", "Max", age.max()],
        ["Age", "Missing", age.isna().sum()],
        ["Weight (kg)", "N", weight.notna().sum()],
        ["Weight (kg)", "Mean", weight.mean()],
        ["Weight (kg)", "Std", weight.std()],
        ["Weight (kg)", "Min", weight.min()],
        ["Weight (kg)", "Median", weight.median()],
        ["Weight (kg)", "Max", weight.max()],
        ["Weight (kg)", "Missing", weight.isna().sum()],
        ["Weight (lb)", "Mean", weight_lb.mean()],
        ["Weight (lb)", "Std", weight_lb.std()],
    ]

    sex = (
        demograph["sex"]
        .map(SEX_LABELS)
        .fillna("Unknown")
    )
    sex_counts = sex.value_counts()
    sex_total = int(sex_counts.sum())

    for label, count in sex_counts.items():
        rows.append(["Sex", label, int(count)])
        rows.append([
            "Sex",
            f"{label} percent",
            count / sex_total * 100.0,
        ])

    rows.append([
        "Sex",
        "Missing",
        int(demograph["sex"].isna().sum()),
    ])

    return pd.DataFrame(
        rows,
        columns=["variable", "statistic", "value"],
    )


def calculate_group_statistics(merged, group_column):
    stats = (
        merged.groupby(
            group_column,
            observed=False,
            dropna=False,
        )
        .agg(
            total_n=("ecg_id", "size"),
            error_n=("error", "sum"),
        )
        .reset_index()
    )

    stats["correct_n"] = (
        stats["total_n"] - stats["error_n"]
    )

    stats["error_rate_pct"] = (
        stats["error_n"] / stats["total_n"] * 100.0
    )

    stats["accuracy_pct"] = (
        stats["correct_n"] / stats["total_n"] * 100.0
    )

    total_errors = int(merged["error"].sum())

    if total_errors > 0:
        stats["error_share_pct"] = (
            stats["error_n"] / total_errors * 100.0
        )
    else:
        stats["error_share_pct"] = 0.0

    return stats


def show_group_plot(
    fold_name,
    stats,
    category_column,
    title,
    x_label,
    output_dir,
):
    """Display age/weight results with counts and within-group error rate."""
    data = stats.dropna(subset=[category_column]).copy()

    if data.empty:
        return

    x = np.arange(len(data))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(11, 6))

    ax1.bar(
        x - width / 2,
        data["total_n"],
        width,
        label="All testing ECGs",
    )

    ax1.bar(
        x + width / 2,
        data["error_n"],
        width,
        label="Incorrect predictions",
    )

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Number of ECGs")
    ax1.set_xticks(x)

    rotation = 45 if "group" in category_column else 0
    alignment = "right" if rotation else "center"

    ax1.set_xticklabels(
        data[category_column].astype(str),
        rotation=rotation,
        ha=alignment,
    )

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        data["error_rate_pct"],
        marker="o",
        linewidth=2,
        label="Error rate",
    )

    ax2.set_ylabel("Error rate within group (%)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
    )

    ax1.set_title(title)
    fig.tight_layout()
    
    # 保存图片
    save_figure(fig, title, output_dir)


def show_cross_fold_age_weight_histograms(
    cross_fold_age_stats,
    cross_fold_weight_stats,
    output_dir,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 12),
    )

    # ------------------------------------------------------------------------
    # Age
    # ------------------------------------------------------------------------
    age_data = cross_fold_age_stats.dropna(
        subset=["age_group"]
    ).copy()

    if not age_data.empty:
        x_age = np.arange(len(age_data))
        width = 0.38

        axes[0].bar(
            x_age - width / 2,
            age_data["total_n"],
            width,
            label="All testing ECGs",
        )

        axes[0].bar(
            x_age + width / 2,
            age_data["error_n"],
            width,
            label="Incorrect predictions",
        )

        axes[0].set_xlabel("Age group")
        axes[0].set_ylabel("Number of ECGs")
        axes[0].set_xticks(x_age)
        axes[0].set_xticklabels(
            age_data["age_group"].astype(str),
            rotation=45,
            ha="right",
        )

        age_ax2 = axes[0].twinx()
        age_ax2.plot(
            x_age,
            age_data["error_rate_pct"],
            marker="o",
            linewidth=2,
            label="Relative error rate",
        )
        age_ax2.set_ylabel(
            "Relative error rate within age group (%)"
        )
        age_ax2.set_ylim(
            0,
            max(
                100.0,
                float(
                    age_data["error_rate_pct"].max()
                ) * 1.15,
            ),
        )

        axes[0].set_title(
            "All five folds: age-related prediction errors"
        )

        handles1, labels1 = (
            axes[0].get_legend_handles_labels()
        )
        handles2, labels2 = (
            age_ax2.get_legend_handles_labels()
        )

        axes[0].legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper left",
        )

    # ------------------------------------------------------------------------
    # Weight
    # ------------------------------------------------------------------------
    weight_data = cross_fold_weight_stats.dropna(
        subset=["weight_group"]
    ).copy()

    if not weight_data.empty:
        x_weight = np.arange(len(weight_data))
        width = 0.38

        axes[1].bar(
            x_weight - width / 2,
            weight_data["total_n"],
            width,
            label="All testing ECGs",
        )

        axes[1].bar(
            x_weight + width / 2,
            weight_data["error_n"],
            width,
            label="Incorrect predictions",
        )

        axes[1].set_xlabel("Weight group")
        axes[1].set_ylabel("Number of ECGs")
        axes[1].set_xticks(x_weight)
        axes[1].set_xticklabels(
            weight_data["weight_group"].astype(str),
            rotation=45,
            ha="right",
        )

        weight_ax2 = axes[1].twinx()
        weight_ax2.plot(
            x_weight,
            weight_data["error_rate_pct"],
            marker="o",
            linewidth=2,
            label="Relative error rate",
        )
        weight_ax2.set_ylabel(
            "Relative error rate within weight group (%)"
        )
        weight_ax2.set_ylim(
            0,
            max(
                100.0,
                float(
                    weight_data["error_rate_pct"].max()
                ) * 1.15,
            ),
        )

        axes[1].set_title(
            "All five folds: weight-related prediction errors"
        )

        handles1, labels1 = (
            axes[1].get_legend_handles_labels()
        )
        handles2, labels2 = (
            weight_ax2.get_legend_handles_labels()
        )

        axes[1].legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper left",
        )

    fig_title = "Cross-five-fold demographic prediction error distributions"
    fig.suptitle(
        fig_title,
        fontsize=16,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    # save image
    save_figure(fig, fig_title, output_dir)

    return fig


def show_gender_histograms_all_folds(
    fold_outputs,
    output_dir,
):
    fold_names = list(fold_outputs.keys())

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 14),
    )

    axes = np.asarray(axes).reshape(-1)

    for index, fold_name in enumerate(fold_names):
        ax = axes[index]

        data = fold_outputs[fold_name]["gender"].copy()
        data = data[
            data["sex_label"].isin(
                ["Male", "Female"]
            )
        ]

        if data.empty:
            ax.set_title(f"{fold_name}: Gender")
            ax.axis("off")
            continue

        x = np.arange(len(data))
        width = 0.35

        ax.bar(
            x - width / 2,
            data["total_n"],
            width,
            label="All testing ECGs",
        )

        ax.bar(
            x + width / 2,
            data["error_n"],
            width,
            label="Incorrect predictions",
        )

        ax.set_title(
            f"{fold_name}: Gender-related prediction errors"
        )
        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of ECGs")
        ax.set_xticks(x)
        ax.set_xticklabels(
            data["sex_label"].astype(str)
        )
        ax.legend()

    for index in range(len(fold_names), len(axes)):
        axes[index].axis("off")

    fig_title = "Gender-related prediction errors across all five folds"
    fig.suptitle(
        fig_title,
        fontsize=16,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
    )
    
    # save image
    save_figure(fig, fig_title, output_dir)
    
    return fig


def build_cross_fold_age_weight_statistics(
    fold_outputs,
):
    merged_frames = [
        output["merged"].copy()
        for output in fold_outputs.values()
    ]

    combined = pd.concat(
        merged_frames,
        ignore_index=True,
    )

    age_stats = calculate_group_statistics(
        combined,
        "age_group",
    )

    weight_stats = calculate_group_statistics(
        combined,
        "weight_group",
    )

    return age_stats, weight_stats


def write_excel(
    output_path,
    testing_summary,
    demo_overall,
    fold_outputs,
    cross_fold_age_stats,
    cross_fold_weight_stats,
):
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".xlsx":
        output_path = output_path.with_suffix(".xlsx")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="xlsxwriter",
    ) as writer:

        workbook = writer.book

        header_format = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#1F4E78",
            "align": "center",
            "valign": "vcenter",
        })

        percent_format = workbook.add_format({
            "num_format": "0.00",
        })

        probability_format = workbook.add_format({
            "num_format": "0.000000",
        })

        integer_format = workbook.add_format({
            "num_format": "0",
        })

        def write_dataframe(dataframe, sheet_name):
            dataframe.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
            )

            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.hide_gridlines(2)

            for col_idx, column_name in enumerate(dataframe.columns):
                worksheet.write(
                    0,
                    col_idx,
                    column_name,
                    header_format,
                )

                if len(dataframe) > 0:
                    max_length = max(
                        len(str(column_name)),
                        int(
                            dataframe[column_name]
                            .astype(str)
                            .map(len)
                            .max()
                        ),
                    )
                else:
                    max_length = len(str(column_name))

                worksheet.set_column(
                    col_idx,
                    col_idx,
                    min(max(max_length + 2, 10), 35),
                )

            for column_name in [
                "accuracy",
                "sensitivity",
                "specificity",
                "error_rate_pct",
                "accuracy_pct",
                "error_share_pct",
            ]:
                if column_name in dataframe.columns:
                    col_idx = dataframe.columns.get_loc(column_name)
                    worksheet.set_column(
                        col_idx,
                        col_idx,
                        16,
                        percent_format,
                    )

            for column_name in [
                "auc",
                "probability",
                "threshold",
            ]:
                if column_name in dataframe.columns:
                    col_idx = dataframe.columns.get_loc(column_name)
                    worksheet.set_column(
                        col_idx,
                        col_idx,
                        16,
                        probability_format,
                    )

            for column_name in [
                "n_test",
                "n_error",
                "total_n",
                "error_n",
                "correct_n",
                "TN",
                "FP",
                "FN",
                "TP",
            ]:
                if column_name in dataframe.columns:
                    col_idx = dataframe.columns.get_loc(column_name)
                    worksheet.set_column(
                        col_idx,
                        col_idx,
                        12,
                        integer_format,
                    )

        write_dataframe(
            testing_summary,
            "Overall_Testing",
        )

        write_dataframe(
            demo_overall,
            "Demographic_Overall",
        )

        for fold_name, output in fold_outputs.items():
            write_dataframe(
                output["merged"],
                f"{fold_name}_Testing"[:31],
            )
            write_dataframe(
                output["age"],
                f"{fold_name}_Age_Error"[:31],
            )
            write_dataframe(
                output["gender"],
                f"{fold_name}_Gender_Error"[:31],
            )
            write_dataframe(
                output["weight"],
                f"{fold_name}_Weight_Error"[:31],
            )

        write_dataframe(
            cross_fold_age_stats,
            "All_Folds_Age_Error",
        )

        write_dataframe(
            cross_fold_weight_stats,
            "All_Folds_Weight_Error",
        )

    return str(output_path.resolve())


def main():
    # argument
    parser = argparse.ArgumentParser(description="ECG Demographic Analysis and Plotting Tool")
    parser.add_argument(
        "--testing-results",
        type=str,
        default=DEFAULT_TESTING_RESULTS_PATH,
        help="Path to the kfold testing results Excel file."
    )
    parser.add_argument(
        "--demograph",
        type=str,
        default=DEFAULT_DEMOGRAPH_PATH,
        help="Path to the simplified demographic CSV file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to the output Excel analysis report."
    )
    args = parser.parse_args()

    testing_results_path = args.testing_results
    demograph_path = args.demograph
    output_dir = Path(args.output)

    testing_results, demograph = load_input_data(testing_results_path, demograph_path)

    testing_summary = build_testing_summary(
        testing_results
    )

    print("\n" + "=" * 100)
    print("OVERALL TESTING RESULTS")
    print("=" * 100)
    print(
        testing_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    demo_overall = build_demographic_overall(
        demograph
    )

    print("\n" + "=" * 100)
    print("OVERALL DEMOGRAPHIC DISTRIBUTION")
    print("=" * 100)
    print(demo_overall.to_string(index=False))

    fold_outputs = {}

    for fold_name, testing_frame in testing_results.items():
        merged = add_demographics(
            testing_frame,
            demograph,
        )

        age_stats = calculate_group_statistics(
            merged,
            "age_group",
        )

        gender_stats = calculate_group_statistics(
            merged,
            "sex_label",
        )

        weight_stats = calculate_group_statistics(
            merged,
            "weight_group",
        )

        fold_outputs[fold_name] = {
            "merged": merged,
            "age": age_stats,
            "gender": gender_stats,
            "weight": weight_stats,
        }

        print("\n" + "=" * 100)
        print(f"{fold_name}: AGE ERROR")
        print("=" * 100)
        print(age_stats.to_string(index=False))

        print("\n" + "=" * 100)
        print(f"{fold_name}: GENDER ERROR")
        print("=" * 100)
        print(gender_stats.to_string(index=False))

        print("\n" + "=" * 100)
        print(f"{fold_name}: WEIGHT ERROR")
        print("=" * 100)
        print(weight_stats.to_string(index=False))

        # Keep the existing age and weight plots unchanged.
        show_group_plot(
            fold_name,
            age_stats,
            "age_group",
            f"{fold_name}: Age-related prediction errors",
            "Age group",
            output_dir,
        )

        show_group_plot(
            fold_name,
            weight_stats,
            "weight_group",
            f"{fold_name}: Weight-related prediction errors",
            "Weight group",
            output_dir,
        )

    # Cross-five-fold age/weight statistics only.
    cross_fold_age_stats, cross_fold_weight_stats = (
        build_cross_fold_age_weight_statistics(
            fold_outputs
        )
    )

    print(" " + "=" * 100)
    print("ALL FIVE FOLDS: AGE ERROR STATISTICS")
    print("=" * 100)
    print(
        cross_fold_age_stats.to_string(
            index=False
        )
    )

    print(" " + "=" * 100)
    print("ALL FIVE FOLDS: WEIGHT ERROR STATISTICS")
    print("=" * 100)
    print(
        cross_fold_weight_stats.to_string(
            index=False
        )
    )

    # One combined figure for all five gender histograms.
    show_gender_histograms_all_folds(
        fold_outputs,
        output_dir,
    )

    # Cross-five-fold age and weight histograms.
    show_cross_fold_age_weight_histograms(
        cross_fold_age_stats,
        cross_fold_weight_stats,
        output_dir,
    )

    output_path = write_excel(
        output_dir,
        testing_summary,
        demo_overall,
        fold_outputs,
        cross_fold_age_stats,
        cross_fold_weight_stats,
    )

    print("\n" + "=" * 100)
    print("Saved one Excel workbook:")
    print(output_path)
    print("=" * 100)

    plt.show()


if __name__ == "__main__":
    main()
