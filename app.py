import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns 

# Import the imp backend ML model connection functions 
from src.feature_engineering import compute_features
from src.prediction import analyze_account
import pandas as pd 


# Set the page configuration (Tab Configuration)
st.set_page_config(
    page_title = "Fake Account Detection Dashboard",
    page_icon = "🔍",
    layout = "wide"
)

# Side bar title 
st.sidebar.title("📌 Navigation")

# Side bar menu
page = st.sidebar.radio(
    "Go to",
    ["Home", "Account Analyzer", "Behavior Insight", "Model Performance"]
)

# Page Routing
if page == "Home":
    st.title("🏠 Fake Account Detection & Risk Analysis Dashboard")

    st.write(
    """
    Social media platforms often face the problem of **mass-created bot accounts**.
    These accounts are usually generated automatically in large numbers and may be used for
    spam, fake engagement, misinformation campaigns, or manipulation of follower counts.

    This system analyzes account behavior and profile characteristics to identify
    **suspicious or fake accounts**.

    The analysis considers factors such as:

    • Username structure  
    • Profile completeness  
    • Posting activity  
    • Follower / following patterns  
    """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("🤖 Supervised Machine Learning Models")

        st.write(
        """
        The following classification models were trained to predict the
        **probability that an account is fake**:

        • Logistic Regression  
        • K-Nearest Neighbors (KNN)  
        • Support Vector Machine (SVC)  
        • Decision Tree Classifier  
        • Random Forest Classifier  
        • XGBoost
        """
        )

    with col2:

        st.subheader("📊 Behavioral Clustering")

        st.write(
        """
        Clustering algorithms were used to identify **behavioral patterns
        among different types of users**.

        Algorithms used:

        • K-Means Clustering  
        • Hierarchical Clustering

        These models group accounts into clusters such as:

        • Moderate Active Users  
        • Highly Active Users  
        • Suspicious Users
        """
        )

    with col3:

        st.subheader("⚠️ Anomaly Detection")

        st.write(
        """
        An anomaly detection algorithm was used to detect accounts
        that behave very differently from typical users.

        Algorithm used:

        • Isolation Forest

        This helps identify **unusual or suspicious account activity**
        that may indicate automated or bot behavior.
        """
        )


elif page == "Account Analyzer":
    st.title("🧠 Account Analyzer")

    st.subheader("Account Information")

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username")

    with col2:
        fullname = st.text_input("Full Name")
    
    # To archieve full page width and have vast area 
    bio = st.text_area("Bio / Description")

    col3, col4 = st.columns(2)
    with col3:
        posts = st.number_input("Number of Posts", min_value=0)
        follows = st.number_input("Number of Following", min_value = 0)
        external_url = st.selectbox("External URL", ["Yes", "No"])

    with col4:
        followers = st.number_input("Number of Followers", min_value=0)
        profile_pic = st.selectbox("Profile Picture", ["Yes", "No"])
        private = st.selectbox("Private Account", ["Yes", "No"])

    analyze_button = st.button("🔍 Analyze Account")

    # Backend API connection 
    if analyze_button:

        # Create the raw input
        raw_input = {
            "username": username,
            "fullname": fullname,
            "bio": bio,
            "posts": posts,
            "followers": followers,
            "follows": follows,
            "profile_pic": 1 if profile_pic == "Yes" else 0,
            "external_url": 1 if external_url == "Yes" else 0,
            "private": 1 if private == "Yes" else 0
        }

        # Create the engineerd model required features 
        features = compute_features(raw_input)

        # Converts features into the pandas df
        df = pd.DataFrame([features])

        # Gain the model predictions 
        result = analyze_account(df)

        # Extract the results 
        prob = result["fake_probability"]
        prediction = result["prediction"]
        cluster = result["cluster"]
        anomaly = result["anomaly"]

        # Show the result section 
        st.divider()
        st.subheader("📊 Account Risk Analysis")

        # ----------- Risk Level Logic -----------
        if prob > 0.7:
            risk = "High"
            risk_color = "error"
        elif prob > 0.4:
            risk = "Moderate"
            risk_color = "warning"
        else:
            risk = "Low"
            risk_color = "success"

        # ----------- Cluster Mapping -----------
        cluster_map = {
            0: "Moderate Active Users",
            1: "Highly Active Users",
            2: "Suspicious Users"
        }
        cluster_name = cluster_map.get(cluster)

        # ----------- Top Metrics Row -----------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fake Probability", f"{prob:.2f}")

        with col2:
            st.metric("Risk Level", risk)

        with col3:
            st.metric("Behavior Category", cluster_name)

        # ----------- Anomaly Status (Full Width) -----------
        st.markdown("<br>", unsafe_allow_html=True)

        if anomaly == -1:
            anomaly_status = "Anomalous Behavior"
            st.error("⚠️ Anomaly detected in account behavior")
        else:
            anomaly_status = "Normal Behavior"
            st.success("✅ Account behavior appears normal")

        # ----------- Explanation Section -----------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔍 Behavioral Explanation")

        if cluster == 0:
            st.info("""
            This account belongs to the **Moderate Active Users cluster**.

            - Profile usually contains a profile picture
            - Moderate posting activity
            - Balanced follower–following ratio
            - Normal account description length
            """)

        elif cluster == 1:
            st.success("""
            This account belongs to the **Highly Active Users cluster**.

            - Higher number of posts and followers
            - Detailed profile descriptions
            - Often includes external links
            - High engagement activity
            """)

        elif cluster == 2:
            st.warning("""
            This account belongs to the **Suspicious Users cluster**.

            - Username contains many numeric characters
            - Profile picture often missing
            - Very short or empty description
            - Extremely low posting activity
            """)

elif page == "Behavior Insight":
    st.title("📊 Behavior Insights")
    
    # Load the dataset 
    df = pd.read_csv("dataset/clustered_accounts.csv").copy()
    df_new = pd.read_csv("dataset/Instagram_fake_profile_dataset.csv")
    
    # Part 1 : Distribution graphs
    col1, col2 = st.columns(2)
    with col1:
        # Cluster Distribution chart 
        cluster_map = {
            0: "Moderate Users",
            1: "Highly Active Users",
            2: "Suspicious Users"
        }
        df["Clusters_Type"] = df["Clusters"].map(cluster_map)
        
        st.subheader("Cluster Distribution")
 
        fig, ax = plt.subplots(figsize = (6, 4))
        sns.countplot(
            data = df,
            x = "Clusters_Type",
            palette = ["green", "blue", "red"],
            order = ["Moderate Users", "Highly Active Users", "Suspicious Users"]
        )

        ax.set_title("User Behavior Clusters")
        ax.set_xlabel("Cluster Type")
        ax.set_ylabel("Number of Accounts")

        plt.xticks(rotation = 10)
        st.pyplot(fig)
    
    with col2:
        # Anomaly Distribution chart 
        anomaly_map = {
            -1: "Anomalous User",
             1: "Normal User"
        }
        df["Anomalies_Type"] = df["Anomalies"].map(anomaly_map) 

        st.subheader("Anomaly Detection Overview")

        fig, ax = plt.subplots(figsize = (6, 4))
        sns.countplot(
            data = df,
            x = "Anomalies_Type",
            palette = ["red", "blue"],
            order = ["Anomalous User", "Normal User"]
        )

        ax.set_title("Anomaly Distribution")
        ax.set_xlabel("Anomaly Type")
        ax.set_ylabel("Number of Accounts")
        
        plt.xticks(rotation = 10)
        st.pyplot(fig)
    
    st.markdown("<br>", unsafe_allow_html = True)

    # Part 2 : Important Histograms 
    st.subheader("📈 Key Behavioral Patterns")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1
    sns.histplot(
        ax=axes[0, 0],
        data=df_new,
        x="nums/length username",
        bins=20,
        hue="fake",
        multiple="dodge"
    )
    axes[0, 0].set_title("Username Numeric Ratio")

    # 2
    sns.histplot(
        ax=axes[0, 1],
        data=df_new,
        x="nums/length fullname",
        bins=20,
        hue="fake",
        multiple="dodge"
    )
    axes[0, 1].set_title("Fullname Numeric Ratio")

    # 3
    sns.histplot(
        ax=axes[1, 0],
        data=df_new,
        x="description length",
        bins=20,
        hue="fake",
        multiple="dodge"
    )
    axes[1, 0].set_title("Description Length")

    # 4
    sns.histplot(
        ax=axes[1, 1],
        data=df_new,
        x="#posts",
        bins=20,
        hue="fake",
        multiple="dodge"
    )
    axes[1, 1].set_title("Number of Posts")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html = True)

    # Thrid Part : Strong indicator Section 
    st.subheader("📌 Strong Behavioral Indicator")

    # Prepare data for the plot 
    fig2, ax2 = plt.subplots(figsize=(6, 4))

    plot_data = (
        df_new.groupby(["profile pic", "fake"])
        .size()
        .reset_index(name="Count")
    )

    sns.barplot(
        ax=ax2,
        data=plot_data,
        x="profile pic",
        y="Count",
        hue="fake"
    )

    ax2.set_title("Profile Picture vs Fake Accounts")
    st.pyplot(fig2)
    

elif page == "Model Performance":
    st.title("📈 Model Performance")

    st.subheader("📊 Model Comparison")

    # Create the All models performance dataframe 
    data = {
        "Model": [
            "Logistic Regression",
            "KNN",
            "Decision Tree",
            "Random Forest",
            "XGBoost"
        ],
        "Train Accuracy": [0.968, 0.992, 1.0, 0.996, 1.0],
        "Test Accuracy": [0.967, 0.988, 0.975, 0.984, 0.99],
        "Precision": [0.971, 0.996, 0.976, 0.990, 0.996],
        "Recall": [0.962, 0.98, 0.974, 0.978, 0.984],
        "F1 Score": [0.967, 0.988, 0.975, 0.984, 0.99]
    }
    df_perf = pd.DataFrame(data)

    # Show the comparison table 
    st.dataframe(df_perf)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Best model highlighting 
    st.subheader("🏆 Best Model")
    st.success("XGBoost achieved the highest performance with 99% accuracy and excellent F1-score.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🧠 Key Insights")
    st.write("""
    - Ensemble models outperform single models due to better handling of non-linear patterns.
    - Logistic Regression provides strong baseline performance with good interpretability.
    - KNN performs very well but is computationally expensive.
    - Tree-based models show slight overfitting (high train accuracy).
    - XGBoost achieves the best balance between accuracy and generalization.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("⚖️ Model Selection Decision")
    st.write("""
    Although **XGBoost achieved the highest performance**, the deployed model in this application is **KNN**.

    Reasoning:

    - All models showed very high performance (above 96% accuracy), so performance difference is minimal.
    - KNN was chosen to better understand:
    - Feature scaling impact
    - Distance-based learning behavior
    - Data preprocessing importance (log transformation + standardization)
    - This helped in building a strong foundation in ML pipeline design.

    However, in real-world production systems:
    - **XGBoost or ensemble models are generally preferred**
    - Because they handle complex patterns better and scale efficiently.

    This project demonstrates both:
    - **Practical learning (KNN implementation)**
    - **Optimal model understanding (XGBoost performance)**
    """)
    st.info("🚀 Future Improvement: Replace KNN with XGBoost for production deployment.")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Model Accuracy Comparision 
    st.subheader("📉 Model Accuracy Comparison")
    st.bar_chart(df_perf.set_index("Model")["Test Accuracy"])
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Conclusion Part
    st.subheader("📌 Final Conclusion")
    st.write("""
    This system combines supervised learning, clustering, and anomaly detection 
    to provide a comprehensive fake account detection and risk analysis solution.

    It not only predicts whether an account is fake, but also explains behavioral patterns 
    and detects unusual activity, making it a powerful real-world applicable system.
    """)
