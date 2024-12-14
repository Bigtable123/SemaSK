# SemaSK: A Semantics Spatial Keyword Queries with LLMs

## Introduction
Existing spatial keyword querying methods have focused primarily on efficiency and often involve proposals for index structures for efficient query processing. In these studies, due to challenges in measuring the semantic relevance of textual data, query constraints on the textual attributes are largely treated as a keyword matching process, ignoring richer query and data semantics. To advance the semantic aspects, we propose a system named SemaSK that exploits the semantic capabilities of large language models to retrieve geo-textual objects that are more semantically relevant to a query. 

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. **Install Required Packages**

   Ensure you have Python installed. Then, install the necessary packages using:

   ```bash
   pip install -r requirements.txt

   ```

## Setup and Running the Project

1. **Download the Yelp Dataset**

   - Visit the [Yelp Dataset](https://www.yelp.com/dataset) website and download the dataset.
   - Extract the dataset to a preferred directory on your machine.

2. **Generate the Dataset**

   - Open `yelp.ipynb` in Jupyter Notebook.
   - Update the data file paths at the beginning of the notebook to point to the location where you extracted the Yelp dataset, and replace the placeholder OpenAI API key with your actual OpenAI API key.
   - Run all cells in the notebook to generate the processed dataset.

3. **Run the Demo**

   - Execute `demo.py` to run the entire project:

     ```bash
     python demo.py
     ```

## Project Structure

- `yelp.ipynb`: Jupyter Notebook to process and generate the dataset from the raw Yelp data.
- `demo.py`: Main script to execute the project workflow.
- `requirements.txt`: Lists all the Python packages required to run the project.
- `query/`: Contains scripts for generating the initial test set queries.
- `test/`: Includes code for testing different systems within the project.

## Contact

If you have any questions or encounter any issues, please contact:
📧 **Zesong Zhang**  
zesongz@student.unimelb.edu.au

