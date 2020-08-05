import matplotlib.pyplot as plt
import seaborn as sns

def count_of_appointment(plot_name,x,y,xlabel,ylabel,alpha):  
    plt.figure(figsize=(15,10))
    plt.scatter(x,y,alpha=alpha)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_name)
    plt.savefig('../reports/figures/' + plot_name + '.png')
    
def age_related_diseases(df,x,y,xlabel,ylabel,plot_name):
    plt.figure(figsize=(20,10))
    sns.lineplot(x=x, y =y,hue='Gender',data=df,err_style='band')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_name)
    plt.savefig('../reports/figures/' + plot_name + '.png')

    
 