# Hardware Accelerator Assignment 1:  K Means Clustering on High Dimensional Data

## Faculty of Engineering and Computer Science - Shahid Beheshti University  

### Academic Year: 2024-2025 | Spring Semester  

**Submission Format:** A `zip` file containing the required components as mentioned in the description.  

---

## Objective  
This assignment aims to implement and understand the **KMeans** algorithm and data clustering.  

**Note:** The implementation of **KMeans** must be done **without** using pre-built libraries. Only basic programming language features should be used. However, **Numpy**, **Pandas**, **Matplotlib**, and **Seaborn** are allowed solely for data reading and visualization.  

---

## KMeans Algorithm Explanation  
The **KMeans** algorithm is an **unsupervised learning** method that clusters data into a predefined number of groups, maximizing intra-cluster similarity while minimizing inter-cluster differences.  

### Steps of the KMeans Algorithm  
1. **Initialize** `k` cluster centers randomly.  
2. **Assign** each data point to the nearest cluster center.  
3. **Update** the cluster centers by computing the mean of all assigned points.  
4. **Repeat** the process until the cluster centers no longer change or a specific iteration limit is reached.  

---

## Dataset  
The dataset used in this assignment contains information about **high schools in Tehran Province**, including educational and financial characteristics.  

**File Name:** `highschool.csv`  

**Dataset Features:**  
- Number of applications  
- Number of admissions  
- Number of enrolled students  
- Percentage of top 10% students in high school  
- Percentage of top 25% students in high school  
- Total full-time students  
- Number of part-time students (if applicable)  
- Annual tuition for private schools  
- Boarding and meal costs for boarding schools  
- Approximate cost of textbooks  
- Estimated personal student expenses  
- Percentage of teachers with a PhD  
- Percentage of teachers with a Master’s degree  
- Student-to-teacher ratio  
- Percentage of alumni supporting the school  
- Educational cost per student
- High school Graduation rate

---

## Requirements  
- **(A)** The number of clusters `k` must be determined carefully using common selection methods.  
- **(B)** The goal is to minimize the number of iterations. The algorithm should be executed with various iteration limits, and for each case, the cluster labels and execution time should be reported.  
- **(C)** To analyze clustering results, tables with the **mean and standard deviation** of each feature within clusters should be provided. Also, various **visualizations** should be included, such as:  
  - **Scatter Plot**  
  - **Pair Plot**  
  - **Strip Plot**  
  - **Swarm Plot**  
- **(D)** The final output should be stored in a **CSV file**, where each row includes the cluster label and the final cluster centroids.  

---

## Analysis and Questions  
Please select and answer **two** of the following questions:  

1. Do schools with a lower student-to-teacher ratio have a higher graduation rate? Provide a statistical analysis and support your findings with visualizations.  
2. Do schools with stricter admission rates achieve higher average test scores? Provide a statistical analysis and support your findings with visualizations.  
3. Do schools with a higher number of admitted students have a higher graduation rate? Provide a statistical analysis and support your findings with visualizations.  

---

## Data Preprocessing Notes  
- **Raw values of** `applications`, `admissions`, and `enrolled students` **do not provide meaningful insights alone.** New features such as **acceptance rate** and **enrollment rate** should be calculated.  
- **Feature values vary significantly in scale, requiring normalization** before applying KMeans.  

---

## References  
- [U.S. News and World Report’s College Data](https://www.kaggle.com/datasets/flyingwombat/us-news-and-world-reports-college-data)  
- [Notebook](https://www.kaggle.com/code/ellecf/visualizing-multidimensional-clusters)  
