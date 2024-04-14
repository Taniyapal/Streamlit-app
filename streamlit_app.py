import streamlit
import streamlit as st
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import scipy.stats as stats
import statsmodels.stats.api as sms
import numpy as np

st.title("A/B Testing")

uploaded_file = st.file_uploader("Upload your dataset here")
if uploaded_file is not None:
    file = pd.read_excel(uploaded_file, "ABTest")

    def abtesting(data):
        z = data[data["Group"] == "Control"]["Visitors"] / data["Clicks"]
        w = data[data["Group"] == "Experiment"]["Visitors"] / data["Clicks"]
        z = z.fillna(z.median())
        w = w.fillna(w.median())
        data["Converted"] = np.where(z < w, 1, 0)

        control_sample = data[data["Group"] == "Control"].sample(
            n=5000, random_state=12
        )
        treatment_sample = data[data["Group"] == "Experiment"].sample(
            n=5000, random_state=12
        )
        ab_test = pd.concat([control_sample, treatment_sample], axis=0)
        ab_test.reset_index(drop=True, inplace=True)

        std_dev = lambda x: np.std(x, ddof=0)
        std_error = lambda x: stats.sem(x, ddof=0)
        conversion_rate = ab_test.groupby("Group")["Converted"].agg(
            [np.mean, std_dev, std_error]
        )
        conversion_rate.columns = ["conversion_rate", "std_deviation", "std_error"]
        st.write(conversion_rate)
        print("*" * 90)

        control_results = ab_test[ab_test["Group"] == "Control"]["Converted"]
        treatment_results = ab_test[ab_test["Group"] == "Experiment"]["Converted"]

        num_control = control_results.count()
        num_treatment = treatment_results.count()
        successes = [control_results.sum(), treatment_results.sum()]
        nobs = [num_control, num_treatment]

        z_stat, pval = proportions_ztest(successes, nobs=nobs)
        (lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(
            successes, nobs=nobs, alpha=0.05
        )

        st.write(f"Z Statistic - {z_stat:.2f}")
        st.write(f"P-Value - {pval:.3f}")
        st.write(f"CI 95% for control group - [{lower_con:.3f}, {upper_con:.3f}]")
        st.write(f"CI 95% for treatment group - [{lower_treat:.3f}, {upper_treat:.3f}]")
        st.write("*" * 90)

        z_stat, pval = proportions_ztest(successes, nobs=nobs)
        (lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(
            successes, nobs=nobs, alpha=0.01
        )

        st.write(f"Z Statistic - {z_stat:.2f}")
        st.write(f"P-Value - {pval:.3f}")
        st.write(f"CI 99% for control group - [{lower_con:.3f}, {upper_con:.3f}]")
        st.write(f"CI 99% for treatment group - [{lower_treat:.3f}, {upper_treat:.3f}]")
        st.write("*" * 90)
        z_stat, pval = proportions_ztest(successes, nobs=nobs)
        (lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(
            successes, nobs=nobs, alpha=0.1
        )

        st.write(f"Z Statistic - {z_stat:.2f}")
        st.write(f"P-Value - {pval:.3f}")
        st.write(f"CI 90% for control group - [{lower_con:.3f}, {upper_con:.3f}]")
        st.write(f"CI 90% for treatment group - [{lower_treat:.3f}, {upper_treat:.3f}]")
        st.write("*" * 90)
        if pval > 0.05:
            st.write("Failed to reject null hypothesis. Control group is better.")
        elif pval < 0.05:
            st.write("Reject null hypothesis. Experiment group is better.")
        else:
            st.write("Indeterminate.")

    abtesting(file)
