import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, dtype=str, encoding='iso-8859-1')  # Handle different encoding
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, dtype=str)  # Read all columns as strings initially
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Convert numeric columns to appropriate types
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass  # Keep as string if conversion fails
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def preprocess_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            if df[col].nunique() < 10:
                df[col] = df[col].astype('category')
    return df

def get_available_tasks(df):
    tasks = [
        ("Describe Data", describe_data),
        ("Show Data Types", show_data_types),
        ("Show Missing Values", show_missing_values)
    ]
    
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numeric_columns) >= 1:
        tasks.append(("Show Summary Statistics", show_summary_statistics))
    
    if len(numeric_columns) >= 2:
        tasks.extend([
            ("Correlation Matrix", show_correlation_matrix),
            ("Scatter Plot Matrix", show_scatter_plot_matrix),
            ("Linear Regression", perform_linear_regression)
        ])
    
    if len(categorical_columns) >= 1 and len(numeric_columns) >= 1:
        tasks.append(("Box Plot", show_box_plot))
    
    if len(categorical_columns) >= 2 and len(numeric_columns) >= 1:
        tasks.append(("Pivot Table", create_pivot_table))
    
    if len(categorical_columns) >= 1:
        tasks.append(("Bar Plot", show_bar_plot))
    
    return tasks

def describe_data(df):
    st.write("This feature provides a concise summary of your dataset, including count, mean, standard deviation, minimum, and maximum values for numeric columns, and count and unique values for categorical columns.")
    st.write(df.describe(include='all'))

def show_data_types(df):
    st.write("This feature displays the data type of each column in your dataset, helping you understand the nature of your variables.")
    st.write(df.dtypes)

def show_missing_values(df):
    st.write("This feature shows the number and percentage of missing values in each column of your dataset, helping you identify data quality issues.")
    missing = df.isnull().sum()
    missing_percent = 100 * df.isnull().sum() / len(df)
    missing_table = pd.concat([missing, missing_percent], axis=1, keys=['Missing Values', '% Missing'])
    st.write(missing_table)

def show_summary_statistics(df):
    st.write("This feature provides detailed summary statistics for numeric columns in your dataset, including count, mean, standard deviation, minimum, maximum, and quartile values.")
    st.write(df.describe())

def show_correlation_matrix(df):
    st.write("This feature generates a heatmap showing the correlation between numeric variables in your dataset, helping you identify relationships between variables.")
    with st.spinner('Generating correlation matrix...'):
        numeric_df = df.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
        st.pyplot(fig)

def show_scatter_plot_matrix(df):
    st.write("This feature creates a matrix of scatter plots for all pairs of numeric variables in your dataset, allowing you to visualize relationships between multiple variables simultaneously.")
    with st.spinner('Generating scatter plot matrix...'):
        numeric_df = df.select_dtypes(include=['number'])
        fig = sns.pairplot(numeric_df, height=2.5)
        st.pyplot(fig)


def show_box_plot(df):
    st.write("This feature creates a box plot to visualize the distribution of a numeric variable across different categories, helping you identify differences and outliers.")
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    x_axis = st.selectbox("Select categorical column for X-axis", categorical_columns)
    y_axis = st.selectbox("Select numeric column for Y-axis", numeric_columns)
    
    with st.spinner('Generating box plot...'):
        # Determine the number of categories
        categories = df[x_axis].nunique()
        
        # Adjust figure size based on the number of categories
        fig_width = min(max(10, categories * 0.5), 20)  # Width between 10 and 20
        fig_height = 8
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax)
        
        # Rotate labels and adjust their alignment
        if categories > 10:
            plt.xticks(rotation=90, ha='center')
        elif categories > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Adjust layout and display
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add summary statistics
        st.write("Summary statistics for each group:")
        summary = df.groupby(x_axis)[y_axis].describe()
        st.write(summary)
        
        # Perform ANOVA
        groups = [group for name, group in df.groupby(x_axis)[y_axis]]
        f_statistic, p_value = stats.f_oneway(*groups)
        st.write(f"One-way ANOVA results:")
        st.write(f"F-statistic: {f_statistic:.4f}")
        st.write(f"p-value: {p_value:.4f}")


def create_pivot_table(df):
    st.write("This feature creates a pivot table, allowing you to summarize and aggregate data based on multiple dimensions.")
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    index = st.selectbox("Select index column", categorical_columns)
    columns = st.selectbox("Select column for pivot", [col for col in categorical_columns if col != index])
    values = st.selectbox("Select values column", numeric_columns)
    
    display_option = st.selectbox("Select display option", 
                                  ["Table Only", "Table and Bar Chart", "Table and Line Chart", "Table and Heatmap"])
    
    with st.spinner('Creating pivot table...'):
        pivot_table = df.pivot_table(index=index, columns=columns, values=values, aggfunc='mean', fill_value=0, observed=True)
        st.write(pivot_table)
        
        if display_option != "Table Only":
            # Determine the number of categories
            index_categories = pivot_table.index.nunique()
            column_categories = pivot_table.columns.nunique()
            
            # Adjust figure size based on the number of categories
            fig_width = min(max(10, column_categories * 0.5), 20)  # Width between 10 and 20
            fig_height = min(max(6, index_categories * 0.4), 15)   # Height between 6 and 15
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            
            if display_option == "Table and Bar Chart":
                pivot_table.plot(kind='bar', ax=ax)
            elif display_option == "Table and Line Chart":
                pivot_table.plot(kind='line', ax=ax)
            elif display_option == "Table and Heatmap":
                sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f', ax=ax)
            
            # Rotate labels and adjust their alignment
            if column_categories > 10:
                plt.xticks(rotation=90, ha='center')
            elif column_categories > 5:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig)

def perform_linear_regression(df):
    st.write("This feature performs linear regression analysis, allowing you to model the relationship between a target variable and one or more predictor variables.")
    numeric_columns = df.select_dtypes(include=['number']).columns
    target = st.selectbox("Select target variable", numeric_columns)
    features = st.multiselect("Select features", [col for col in numeric_columns if col != target])
    
    if not features:
        st.warning("Please select at least one feature.")
        return
    
    with st.spinner('Performing linear regression...'):
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression().fit(X_train, y_train)
        
        results = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_})
        st.write("Regression Results:")
        st.write(results)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared Score: {r2:.2f}")
        
        if len(features) == 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
            ax.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
            ax.set_xlabel(features[0])
            ax.set_ylabel(target)
            ax.set_title(f'Linear Regression: {target} vs {features[0]}')
            ax.legend()
            st.pyplot(fig)
        else:
            st.write("Multiple features selected. Showing feature importance:")
            feature_importance = pd.DataFrame({'Feature': features, 'Importance': abs(model.coef_)})
            feature_importance = feature_importance.sort_values('Importance', ascending=True)
            
            # Adjust figure size based on the number of features
            fig_height = max(6, len(features) * 0.4)
            fig, ax = plt.subplots(figsize=(10, fig_height))
            
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
            ax.set_title('Feature Importance')
            ax.set_xlabel('Absolute Coefficient Value')
            plt.tight_layout()
            st.pyplot(fig)


def main():
    st.set_page_config(layout="wide")
    st.title("Enhanced Data Analysis App")

    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            df = preprocess_dataframe(df)
            st.write("Data Preview:")
            st.write(df.head())

            available_tasks = [
                ("Describe Data", describe_data),
                ("Show Data Types", show_data_types),
                ("Show Missing Values", show_missing_values),
                ("Show Summary Statistics", show_summary_statistics),
                ("Correlation Matrix", show_correlation_matrix),
                ("Scatter Plot Matrix", show_scatter_plot_matrix),
                ("Box Plot", show_box_plot),
                ("Pivot Table", create_pivot_table),
                ("Linear Regression", perform_linear_regression),
            ]

            st.sidebar.title("Analysis Options")
            selected_tasks = {}
            for i, (task_name, _) in enumerate(available_tasks):
                selected_tasks[task_name] = st.sidebar.toggle(
                    task_name,
                    key=f"task_{i}"
                )

            for task_name, task_function in available_tasks:
                if selected_tasks[task_name]:
                    st.write(f"## {task_name}")
                    task_function(df)

if __name__ == "__main__":
    main()