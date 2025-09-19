"""
Analyzing Data with Pandas and Visualizing Results with Matplotlib
Assignment Solution - Iris Dataset Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_explore_data():
    """Task 1: Load and Explore the Dataset"""
    print("=" * 60)
    print("TASK 1: LOADING AND EXPLORING THE DATASET")
    print("=" * 60)
    
    try:
        # Load the Iris dataset
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        
        # Display first few rows
        print("\nFirst 5 rows of the dataset:")
        print(df.head())
        
        # Explore dataset structure
        print("\nDataset information:")
        print(df.info())
        
        # Check for missing values
        print("\nMissing values in each column:")
        missing_values = df.isnull().sum()
        print(missing_values)
        
        # Since the Iris dataset is clean, we'll demonstrate cleaning with a hypothetical scenario
        if missing_values.sum() == 0:
            print("\nNo missing values found in the dataset.")
        else:
            # This would execute if there were missing values
            print("\nCleaning dataset by filling missing values...")
            # For numerical columns, fill with mean
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
            
            # For categorical columns, fill with mode
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def basic_data_analysis(df):
    """Task 2: Basic Data Analysis"""
    print("\n" + "=" * 60)
    print("TASK 2: BASIC DATA ANALYSIS")
    print("=" * 60)
    
    # Basic statistics for numerical columns
    print("Basic statistics for numerical columns:")
    print(df.describe())
    
    # Group by species and compute mean for numerical columns
    print("\nMean values by species:")
    species_group = df.groupby('species').mean()
    print(species_group)
    
    # Additional analysis: find the species with the largest sepal length
    max_sepal_length = df.loc[df['sepal length (cm)'].idxmax()]
    print(f"\nThe flower with the largest sepal length is: {max_sepal_length['species']}")
    print(f"Sepal length: {max_sepal_length['sepal length (cm)']} cm")
    
    # Find correlations between features
    print("\nCorrelation matrix (numerical features only):")
    numerical_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numerical_df.corr()
    print(correlation_matrix)
    
    # Interesting finding: strongest correlation
    strongest_corr = correlation_matrix.unstack().sort_values(ascending=False)
    # Remove correlations of 1.0 (self-correlation)
    strongest_corr = strongest_corr[strongest_corr != 1.0]
    print(f"\nStrongest correlation between features: {strongest_corr.index[0]} = {strongest_corr.iloc[0]:.3f}")

def create_visualizations(df):
    """Task 3: Data Visualization"""
    print("\n" + "=" * 60)
    print("TASK 3: DATA VISUALIZATION")
    print("=" * 60)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Iris Dataset Analysis Visualizations', fontsize=16, fontweight='bold')
    
    # 1. Line chart showing trends (using index as pseudo-time)
    # We'll create a line chart showing how sepal length varies across the dataset
    axes[0, 0].plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
    axes[0, 0].plot(df.index, df['petal length (cm)'], label='Petal Length', color='red')
    axes[0, 0].set_title('Trend of Sepal and Petal Length Across Samples')
    axes[0, 0].set_xlabel('Sample Index')
    axes[0, 0].set_ylabel('Length (cm)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bar chart showing comparison of numerical values across categories
    species_means = df.groupby('species').mean()
    x = np.arange(len(species_means.index))
    width = 0.2
    
    axes[0, 1].bar(x - width, species_means['sepal length (cm)'], width, label='Sepal Length')
    axes[0, 1].bar(x, species_means['sepal width (cm)'], width, label='Sepal Width')
    axes[0, 1].bar(x + width, species_means['petal length (cm)'], width, label='Petal Length')
    axes[0, 1].set_title('Average Measurements by Species')
    axes[0, 1].set_xlabel('Species')
    axes[0, 1].set_ylabel('Measurement (cm)')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(species_means.index)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Histogram of a numerical column
    axes[1, 0].hist(df['sepal length (cm)'], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('Distribution of Sepal Length')
    axes[1, 0].set_xlabel('Sepal Length (cm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_sepal = df['sepal length (cm)'].mean()
    median_sepal = df['sepal length (cm)'].median()
    axes[1, 0].axvline(mean_sepal, color='red', linestyle='--', label=f'Mean: {mean_sepal:.2f}')
    axes[1, 0].axvline(median_sepal, color='green', linestyle='--', label=f'Median: {median_sepal:.2f}')
    axes[1, 0].legend()
    
    # 4. Scatter plot to visualize relationship between two numerical columns
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        axes[1, 1].scatter(species_data['sepal length (cm)'], 
                          species_data['petal length (cm)'], 
                          label=species, 
                          alpha=0.7,
                          color=colors[species])
    
    axes[1, 1].set_title('Sepal Length vs Petal Length')
    axes[1, 1].set_xlabel('Sepal Length (cm)')
    axes[1, 1].set_ylabel('Petal Length (cm)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.savefig('iris_analysis_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create an additional visualization: pairplot using seaborn
    print("\nCreating additional visualization: Pairplot of all features...")
    sns.pairplot(df, hue='species', diag_kind='hist', palette='husl')
    plt.suptitle('Pairplot of Iris Dataset Features by Species', y=1.02)
    plt.savefig('iris_pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the analysis"""
    print("ANALYZING DATA WITH PANDAS AND VISUALIZING RESULTS WITH MATPLOTLIB")
    print("Dataset: Iris Flower Dataset")
    print("=" * 60)
    
    # Task 1: Load and explore the dataset
    df = load_and_explore_data()
    
    if df is not None:
        # Task 2: Basic data analysis
        basic_data_analysis(df)
        
        # Task 3: Data visualization
        create_visualizations(df)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Key Findings:")
        print("- The Iris dataset contains measurements for three species of iris flowers")
        print("- Setosa flowers have distinctly different measurements from versicolor and virginica")
        print("- There is a strong correlation between petal length and petal width")
        print("- Virginica flowers tend to have the largest measurements overall")
        print("\nVisualizations have been saved as 'iris_analysis_visualizations.png' and 'iris_pairplot.png'")
    else:
        print("Failed to load dataset. Exiting program.")

if __name__ == "__main__":
    main()
