# PPHA 30537
# Spring 2024
# Homework 3

# YOUR NAME HERE: Abu Bakar Siddique

# YOUR CANVAS NAME HERE: abubakars
# YOUR GITHUB USER NAME HERE: abubakars25

# Due date: Sunday May 5th before midnight
# Write your answers in the space between the questions, and commit/push only
# this file to your repo. Note that there can be a difference between giving a
# "minimally" right answer, and a really good answer, so it can pay to put
# thought into your work.

##################

#NOTE: All of the plots the questions ask for should be saved and committed to
# your repo under the name "q1_1_plot.png" (for 1.1), "q1_2_plot.png" (for 1.2),
# etc. using fig.savefig. If a question calls for more than one plot, name them
# "q1_1a_plot.png", "q1_1b_plot.png",  etc.

# Question 1.1: With the x and y values below, create a plot using only Matplotlib.
# You should plot y1 as a scatter plot and y2 as a line, using different colors
# and a legend.  You can name the data simply "y1" and "y2".  Make sure the
# axis tick labels are legible.  Add a title that reads "HW3 Q1.1".

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

x = pd.date_range(start='1990/1/1', end='1991/12/1', freq='MS')
y1 = np.random.normal(10, 2, len(x))
y2 = [np.sin(v)+10 for v in range(len(x))]

#Data
x = pd.date_range(start='1990/1/1', end='1991/12/1', freq='MS')
y1 = np.random.normal(10, 2, len(x))
y2 = [np.sin(v)+10 for v in range(len(x))]

#Plotting
plt.scatter(x, y1, color = 'green', label = 'y1')
plt.plot(x, y2, color = 'red', label = 'y2')

#Adding title, label and legends
plt.title('HW3 Q1.1', color = 'blue', fontsize = '14')
plt.xlabel('Date', color = 'blue', fontsize = '14')
plt.ylabel('Values', color = 'blue', fontsize = '14')
plt.legend()
plt.xticks(rotation = 45)
plt.tight_layout()

#Saving the figure
plt.savefig('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/q1_plot.png')

plt.show()

# Question 1.2: Using only Matplotlib, reproduce the figure in this repo named
# question_2_figure.png.

#Data
x = 10,19
y = 10,19
x1 = 10,19
y1 = 19,10

#Plotting
plt.plot(x, y, c = 'blue', label='Blue')
plt.plot(x1, y1, c = 'red', label='Red')
plt.legend(loc ="center left")  
plt.title('X marks the spot')


#Saving the figure
plt.savefig('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/question_2_figure.png')

plt.show()

# Question 1.3: Load the mpg.csv file that is in this repo, and create a
# plot that tests the following hypothesis: a car with an engine that has
# a higher displacement (i.e. is bigger) will get worse gas mileage than
# one that has a smaller displacement.  Test the same hypothesis for mpg
# against horsepower and weight.

df = pd.read_csv('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/mpg.csv')
df.head()

#Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (17, 7))

#MPG and Displacement 
ax1.scatter(df['mpg'], df['displacement'], color='blue')
ax1.set_xlabel('MPG', color = 'green', fontsize = '13')
ax1.set_ylabel('Displacement', color = 'green', fontsize = '13')
ax1.set_title('Displacement vs MPG', color = 'red', fontsize = '15')

#The plot shows that lower MPG is associated with higher displacement. MPG has negative 
#relationship with displacement.

#MPG and Horsepower 
ax2.scatter(df['mpg'], df['horsepower'], color='purple')
ax2.set_xlabel('MPG', color = 'green', fontsize = '13')
ax2.set_ylabel('Horsepower', color = 'green', fontsize = '13')
ax2.set_title('Horsepower vs MPG', color = 'red', fontsize = '15')

#The plot shows that lower MPG is associated with higher horsepower. MPG has negative 
#relationship with horsepower.

#MPG and Weight 
ax3.scatter(df['mpg'], df['weight'], color='orange')
ax3.set_xlabel('MPG', color = 'green', fontsize = '13')
ax3.set_ylabel('Weight', color = 'green', fontsize = '13')
ax3.set_title('Weight vs MPG', color = 'red', fontsize = '15')

#The plot shows that lower MPG is associated with higher weight. MPG has negative 
#relationship with weight.

#Saving the figure
fig.savefig('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/question1_3_figure.png')

plt.show()


# Question 1.4: Continuing with the data from question 1.3, create a scatter plot 
# with mpg on the y-axis and cylinders on the x-axis.  Explain what is wrong 
# with this plot with a 1-2 line comment.  Now create a box plot using Seaborn
# that uses cylinders as the groupings on the x-axis, and mpg as the values
# up the y-axis.

fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(df['cylinders'], df['mpg'])
ax.set_xlabel('cylinders')
ax.set_ylabel('mpg')
ax.set_title('Scatter plot of MPG and Cylinders')

#Saving the figure
fig.savefig('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/question1_4a_figure.png')

#Issue with creating scatter plot
#Multiple data points of MPG have same or similar value which is causing overlapping. Thus,  
#it is difficult to discern that how many cars share same cylinders. 

#Creating box plot
fig, ax = plt.subplots(figsize = (10, 6))
sns.boxplot(x = 'cylinders', y = 'mpg', data = df, ax = ax)
ax.set_title('MPG Distribution by Number of Cylinders')
ax.set_xlabel('No. of cylinders')
ax.set_ylabel('Distribution of MPG')

#Saving the figure
fig.savefig('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/question1_4b_figure.png')

# Question 1.5: Continuing with the data from question 1.3, create a two-by-two 
# grid of subplots, where each one has mpg on the y-axis and one of 
# displacement, horsepower, weight, and acceleration on the x-axis.  To clean 
# up this plot:
#   - Remove the y-axis tick labels (the values) on the right two subplots - 
#     the scale of the ticks will already be aligned because the mpg values 
#     are the same in all axis.  
#   - Add a title to the figure (not the subplots) that reads "Changes in MPG"
#   - Add a y-label to the figure (not the subplots) that says "mpg"
#   - Add an x-label to each subplot for the x values
# Finally, use the savefig method to save this figure to your repo.  If any
# labels or values overlap other chart elements, go back and adjust spacing.


# Create a two-by-two grid of subplots
fig, axs = plt.subplots(2, 2, figsize = (14, 10))

# Plot mpg against displacement
axs[0, 0].scatter(df['displacement'], df['mpg'], alpha=0.5, c="red")
axs[0, 0].set_xlabel('Displacement')

# Plot mpg against horsepower
axs[0, 1].scatter(df['horsepower'], df['mpg'], alpha=0.5, c="blue")
axs[0, 1].set_xlabel('Horsepower')
axs[0, 1].get_yaxis().set_visible(False) 

# Plot mpg against weight
axs[1, 0].scatter(df['weight'], df['mpg'], alpha=0.5, c="green")
axs[1, 0].set_xlabel('Weight')

# Plot mpg against acceleration
axs[1, 1].scatter(df['acceleration'], df['mpg'], alpha=0.5, c="orange")
axs[1, 1].set_xlabel('Acceleration')
axs[1, 1].get_yaxis().set_visible(False) 

# Add overall title to the figure
fig.suptitle('Changes in MPG', fontsize=16)

# Add y-label to the figure
fig.text(0, 0.5, 'MPG', va='center', rotation='vertical')

#Saving the figure
fig.savefig('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/question1_5_figure.png')

plt.tight_layout()
plt.show()

# Question 1.6: Are cars from the USA, Japan, or Europe the least fuel
# efficient, on average?  Answer this with a plot and a one-line comment.

mean_mpg_origin = df.groupby('origin')['mpg'].mean()

#Creating the bar chart
plt.figure(figsize=(10, 7))
mean_mpg_origin.plot(kind='bar', color=['purple', 'lightblue', 'lightgreen'])
plt.title('Average Fuel Efficiency by Origin', fontsize=15)
plt.xlabel('Region', fontsize=13)
plt.ylabel('Mean MPG', fontsize=13)
plt.xticks(rotation=0) 

#Saving the figure
plt.savefig('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/question1_6_figure.png')

plt.show()

#Cars from USA are least fuel efficient.

# Question 1.7: Using Seaborn, create a scatter plot of mpg versus displacement,
# while showing dots as different colors depending on the country of origin.
# Explain in a one-line comment what this plot says about the results of 
# question 1.6.

sns.scatterplot(data=df, x='displacement', y='mpg', hue='origin', palette='deep')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.title('MPG vs Displacement by Origin')

#The plot shows that USA cars have lower MPG, but that is explained by their greater displacement. Overall, 
#the pattern of MPG is consistent with displacement across origins. 

# Question 2: The file unemp.csv contains the monthly seasonally-adjusted unemployment
# rates for US states from January 2020 to December 2022. Load it as a dataframe, as well
# as the data from the policy_uncertainty.xlsx file from homework 2 (you do not have to make
# any of the changes to this data that were part of HW2, unless you need to in order to 
# answer the following questions).
#    2.1: Merge both dataframes together

import us as us

#Loading unemployment data
df_unemp = pd.read_csv('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/unemp.csv')
df_unemp.rename(columns={'DATE': 'date', 'STATE': 'state'}, inplace=True)
df_unemp['date']=pd.to_datetime(df_unemp['date'])

#Loading policy data
df_policy = pd.read_excel('/Users/naumanali/Desktop/3rd Quarter/Python/HWs/HW3/policy_uncertainty.xlsx')
df_policy['date'] = pd.to_datetime(df_policy[['year', 'month']].assign(day=1))
df_policy.head()

# Function to convert state abbreviations to state names
def abbrev_to_name(abbrev):
    if abbrev is not None:
        return us.states.lookup(abbrev).name
    else:
        return None

df_unemp['state'] = df_unemp['state'].apply(abbrev_to_name)

#Merging
df_merged = pd.merge(df_unemp, df_policy, on=['state', 'date'])

#    2.2: Calculate the log-first-difference (LFD) of the EPU-C data

#Calculating natural log
df_merged['Log_EPU_Composite'] = np.log(df_merged['EPU_Composite'])

# Calculate the first difference of the log of the EPU-C data
df_merged['LFD_EPU_Composite'] = df_merged['Log_EPU_Composite'].diff()

#    2.2: Select five states and create one Matplotlib figure that shows the unemployment rate
#         and the LFD of EPU-C over time for each state. Save the figure and commit it with 
#         your code.

# Create a figure with subplots for the five states
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(14, 17), sharex=True)

selected_states = ['Alabama', 'Delaware', 'New York', 'Georgia', 'Illinois']

# Plot unemployment rate and LFD of EPU-C for each selected state
for i, state in enumerate(selected_states):
    data_st = df_merged[df_merged['state'] == state]
    axes[i, 0].plot(data_st['date'], data_st['unemp_rate'], label='Unemployment Rate', color='blue')
    axes[i, 0].set_ylabel('Unemployment Rate')
    axes[i, 0].set_title(f'{state} Unemployment Rate')
    
    
    axes[i, 1].plot(data_st['date'], data_st['LFD_EPU_Composite'], label='LFD_EPU-C', color='red')
    axes[i, 1].set_ylabel('LFD-C')
    axes[i, 1].set_title(f'{state} LFD_EPU-C')

#rotates dates in the figure to make them legible
axes[-1, 0].tick_params(labelrotation=45)
axes[-1, 1].tick_params(labelrotation=45)

# Add x-axis label to the last row
axes[-1, 0].set_xlabel('Date')
axes[-1, 1].set_xlabel('Date')

# Adjust layout
plt.tight_layout()
plt.show()


#    2.3: Using statsmodels, regress the unemployment rate on the LFD of EPU-C and fixed
#         effects for states. Include an intercept.

# drop nulls
merged_df_1 = df_merged.dropna()

# Add intercept to the merged data
merged_df_1['intercept'] = np.array([1 for x in range(len(merged_df_1.index))])

import statsmodels.api as sm

# Perform fixed effects regression
fixed_eff_model = sm.OLS(merged_df_1['unemp_rate'], merged_df_1[['LFD_EPU_Composite', 'intercept']])
fixed_eff_results = fixed_eff_model.fit()

#    2.4: Print the summary of the results, and write a 1-3 line comment explaining the basic
#         interpretation of the results (e.g. coefficient, p-value, r-squared), the way you 
#         might in an abstract.

#Regression summary
print(fixed_eff_results.summary())

#The negatice coefficient of LFD_EPU_Composite (-0.1021) suggests that there is negative relationship with
#the unemployment rate. However, the p-value being greater than 0.05 suggests that it is not statistically
#significant. The R-sqaured value is 0.0000 which tells that model does not explain the variation in unemployment
#rate. The overall result depicts that there is further need of refining the model or using a different 
#statistical approach. 