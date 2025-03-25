import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_null_values(data):
    null_values = data.isnull().sum()
    print("Null Values in each column:\n",null_values, flush=True)

def any_duplicate(data):
    print(f"Duplicate Value:\n{data.duplicated().sum()}",flush=True)

def get_columns(data,show=False):
    if show:
        print(data.columns)
    columns = data.columns
    print(f"Total Column Number : {len(columns)}",flush=True)
    return columns

def show_column_type(data):
    print(data.dtypes)

def drop_columns(data,columns):
    print(f"Dropping Columns:{columns}")
    print(f"Before droping columns the shape is : {data.shape}")
    data = data.drop(columns=columns)
    print(f"After droping columns the shape is : {data.shape}")

def detecting_outlier(data,columns,outdir_path=None,filename=None):
    if outdir_path is None:
        raise ValueError("Please provide output directory path or filename")
    
    fig = px.box(data,y=columns)
    fig.update_layout(title="Outlier Detection in Numerical Columns")
    fig.write_image(f"{outdir_path}/{filename}.jpeg")

def visualize_against_disease(data,target_col,outdir_path,filename,histogram=False,Bar=False,Pie=False):
    if histogram:
        fig = px.histogram(data, x=target_col, color='Diagnosis', barmode='overlay', labels={'Diagnosis':'Diagnosis (0=No, 1=Yes)'})
        fig.update_layout(title=f"Distribution of {target_col} Among Patients with and without Alzheimer's")
        fig.write_image(f"{outdir_path}/{filename}.jpeg")
    
    if Bar:
        fig = px.histogram(data, x=target_col, color='Diagnosis', barmode='group', labels={'Diagnosis':'Diagnosis (0=No, 1=Yes)'})
        fig.update_layout(title=f"Distribution of {target_col} Among Patients with and without Alzheimer's")
        fig.write_image(f"{outdir_path}/{filename}.jpeg")
    if Pie:
        target_percentage = data[target_col].value_counts(normalize=True) * 100
        fig = px.pie(values=target_percentage, names=target_percentage.index, title=f'Percentage of {target_col}\'s')
        fig.write_image(f"{outdir_path}/{filename}.jpeg")

def visualize_correlation(data,columns,outdir_path,filename):
    corr_matrix = data[columns].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Matrix')
    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=100, r=100, b=100, t=100, pad=4)
    )
    fig.write_image(f"{outdir_path}/{filename}.jpeg")

def show_cholesterol_visualiation(data,outdir_path,filename):
    cholesterol_columns = ['CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL']
    fig = make_subplots(rows=1, cols=3, subplot_titles=cholesterol_columns)
    for i, column in enumerate(cholesterol_columns, 1):
        fig.add_trace(go.Scatter(x=data[data['Diagnosis'] == 1]['Age'], y=data[data['Diagnosis'] == 1][column], mode='markers', name=column), row=1, col=i)
        fig.update_xaxes(title_text='Age', row=1, col=i)
        fig.update_yaxes(row=1, col=i)
    fig.update_layout(title="Relationship Between Age and Cholesterol Levels Among Patients with Alzheimer's")
    fig.write_image(f"{outdir_path}/{filename}.jpeg")

def plot_target_distribution(data, target_column, outdir_path=None, filename=None, title=None, figsize=(10, 6)):
    if title is None:
        title = f'Distribution of {target_column}'
    
    # টার্গেট ভেরিয়েবলের কাউন্ট গণনা
    target_counts = data[target_column].value_counts().sort_index()
    
    # টার্গেট ভেরিয়েবলের পার্সেন্টেজ গণনা
    target_percent = data[target_column].value_counts(normalize=True).sort_index() * 100
    
    # প্লট তৈরি
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(target_counts.index.astype(str), target_counts.values, color='skyblue')
    
    # বারের উপরে কাউন্ট এবং পার্সেন্টেজ দেখানো
    for i, (count, percent) in enumerate(zip(target_counts.values, target_percent.values)):
        ax.text(i, count + (count * 0.02), f'{count} ({percent:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    # প্লট সাজানো
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel(target_column, fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # যদি টার্গেট ভেরিয়েবল বাইনারি হয় (0/1) তাহলে লেবেল যোগ করা
    if set(target_counts.index) == {0, 1}:
        ax.set_xticklabels(['No Alzheimer\'s (0)', 'Alzheimer\'s (1)'])
    
    plt.tight_layout()
    
    # ছবি সেভ করা
    if outdir_path and filename:
        plt.savefig(f"{outdir_path}/{filename}.jpeg", dpi=300, bbox_inches='tight')
        print(f"Figure saved at: {outdir_path}/{filename}.jpeg")
    
    # plt.show()
    plt.close()

    