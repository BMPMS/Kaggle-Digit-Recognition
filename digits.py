import pandas as pd
import numpy as np

digit = pd.read_csv('digit_train.csv')


#1. Find out the number of columns
print(len(digit.columns))

#2. Initial formatting

for column in digit:
    #deletes all columns where the average value is 0 i.e constant
    if int(np.mean(digit[column])) == 0:
        digit.drop(column, axis=1, inplace=True)
    else:
        if digit[column].nunique()<=2:
            #prints any further constants - there were none
            print(column)



#3. Put the data into a different format for analysis in R.

#a)group by number:
bydigit = digit.groupby('label')

num_dict = [['zero',0],['one',1],['two',2],['three',3],['four',4],['five',5],['six',6],['seven',7],['eight',8],['nine',9]]
x = 0
cols = ['number','column_name', 'avg_pixels']
pixels = pd.DataFrame(columns=cols)
for n in num_dict:
    #b)for each number
    framename = n[0] + '_df'
    framename = bydigit.get_group(n[1])
    #i)print the number of rows for info
    mylist = []
    mylist2 = []
    for column in framename:
        #c)for each column.
        if int(np.mean(framename[column])) < 20:
            #don't included if the average pixel value is less than 20
            mylist.append(column)
        else:
            if column != 'label':
                #add the number, columnname(number only) and average pixels to new dataframe
                mylist2 = [n[0],int(column[5:]), int(np.mean(framename[column]))]
                pixels.loc[len(pixels)]= mylist2
    print(str(len(mylist)) + ' columns deleted with average pixel value under 20 for ' + n[0] + ' total rows for this number/outcome = ' + str(len(framename)))

#sort the new dataframe by number then column name and then export it.
pixels.sort_values(['number','column_name'])
pixels.to_csv('digit_pixels.csv',index=False)
