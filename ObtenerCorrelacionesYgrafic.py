catCols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
j = 0
for i in catCols:
    plt.figure(j)
    sns.barplot(x = i, y = 'Survived', data = train)
    plt.show()
    j+=1
