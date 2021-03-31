
# coding: utf-8

# # <center><font color='orange'>ΑΣΚΗΣΗ 1 </font>| `Επιβλεπόμενη Μάθηση` : `Ταξινόμηση`<center>
# <p style="text-align: right;"><font color='grey'>
# Εργαστήριο Τεχνητής Νοημοσύνης και Μηχανικής Μάθησης
# <br>
# Νευρωνικά Δίκτυα και Ευφυή Υπολογιστικά Συστήματα
# </font>
# <br><br>
# **Ευδοκία Μπαρουξή** (*16586*) | **Τζανακάκης Γιάννης** (*14436*)
# <br>
# <p style="text-align: right;"><font color='grey'>
# Αριθμός Ομάδας: 83
# </font>
# </p>

# # Εισαγωγή
# Σκοπός της παρούσας εργασίας είναι η εξοικείωση με έννοιες και η εφαρμογή μεθόδων της Επιβλεπόμενης Μηχανικής Μάθησης για το **Classification** (Ταξινόμηση, Κατηγοριοποίηση) δεδομένων ως προς ένα σύνολο γνωρισμάτων/μεταβλητών, καθώς και η σύνταξη μιας παρουσίασης της παραπάνω διαδικασίας, με χρήση markdown formatting, στο περιβάλλον του `Jupyter`.
# 
# Για την ανάλυση , θα χρησιμοποιηθούν 2 σύνολα δεδομένων (διαφορετικού μεγέθους), από τη συλλογή του **UCI Machine Learning Repository**, κάτω από το σύνδεσμο: https://archive.ics.uci.edu/ml/datasets/.
# 
# Ο παρακάτω πίνακας περιέχει κάποιες βασικές πληροφορίες για τα 2 datasets που ανατέθηκαν στην ομάδα μας:
# 
# | Όνομα | Πεδίο | Στιγμιότυπα | Μεταβλητές | Πεδίο Τιμών | Κλάσεις | NAs |
# |:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | Ionosphere | Φυσική | 351 | 34 | Πραγματικοί | 2 | Όχι
# | Isolet | Computing | 7797 | 617 | Πραγματικοί | 26 | Όχι 
# Καλούμαστε να βρούμε για κάθε classifier (πέρα από τους dummy) σε κάθε dataset:
# >1. Τη **βέλτιστη αρχιτεκτονική μετασχηματιστών** (στάδια προ-επεξεργασίας).
# >2. Τις **βέλτιστες υπερ-παραμέτρους** μέσω `grid search` και `cross validation`.
# 
# Το Κεφάλαιο 3 αναφέρεται στην ανάλυση του Ionosphere Dataset (small dataset) και το Κεφάλαιο 4 στη μελέτη του Isolet Dataset (big dataset), αντίστοιχα. Αναλυτικότερα τα Κεφάλαια και οι παράγραφοι, αναφέρονται στον πίνακα περιεχομένων της παρουσίασης.

# <div class="alert alert-warning">
#   <strong>Προαπαιτούμενες Βιβλιοθήκες:</strong>
# </div>
# <br>
# Για την αναπαραγωγή του κώδικα της εργασίας, απαιτείται η εγκατάσταση ορισμένων βιβλιοθηκών, μέσω των παρακάτω (ή αντίστοιχων) εντολών:
# 
# >  `pip3 install numpy`
# 
# >  `pip3 install pandas`
# 
# >  `pip3 install seaborn`
# 
# >  `pip3 install matplotlib`
# 
# >  `pip3 install sklearn`
# 
# >  `pip3 install imblearn`
# 

# # Ionosphere Dataset
# 
# Λίγα λόγια για την προέλευση των δεδομένων:
# <div class="alert alert-block alert-info">
# Τα δεδομένα του Ionosphere dataset, συλλέχθηκαν από ένα **τηλεπικοινωνιακό σύστημα** στο Goose Bay, Labrador. Το σύστημα περιελάμβανε **συστοιχία 16 κεραιών υψηλών συχνοτήτων**, με δυνατότητα εκπομπής ισχύος της τάξης των 6.5 kW. Στόχος ήταν τα **ελεύθερα ηλεκτρόνια στην Ιονόσφαιρα** του πλανήτη μας.
# <br> <br>
# Τα εκπομπόμενα σήματα που επέστρεφαν στη γη, κατηγοριοποιούνταν ως "**καλά**", αφού μαρτυρουσαν ύπαρξη δομής στην Ιονόσφαιρα. Αντίθετα, τα σήματα που δεν επέστρεφαν στη γη χαρακτηρίζονταν ως "**κακά**". Η στήλη class του dataframe, με τιμές στο δισύνολο {b,g} αποτελεί την κωδικοποίηση του εν λόγω χαρακτηρισμού. 
# <br> <br>
# Τα σήματα που επέστρεφαν, επεξεργάζονταν μέσω μιας συνάρτησης παλινδρόμησης με ορίσματα τον **χρόνο** και τον **κωδικό** ενός εκπομπόμενου **παλμού**. Υπήρχαν συνολικά 17 διαφορετικοί παλμοί στο σύστημα. Τα στιγμιότυπα (γραμμές) του dataframe περιγράφονται από 2 τιμές για καθέναν από τους 17 παλμούς (2 τιμές για κωδικοποίηση του ηλεκτρομαγνητικού κύματος -μιγαδικό σήμα- που επρόκειτο να εκπέμψει ο παλμός). 
# </div>
#  <br>
# `Παραχωρήθηκε από:`
# 
# >*Space Physics Group, Applied Physics Laboratory, Johns Hopkins University*.
# 
# 
# `Σχετικό Άρθρο:`
# 
# >*Sigillito, V. G., Wing, S. P., Hutton, L. V., \& Baker, K. B. (1989).
# <br>
# Classification of radar returns from the ionosphere using neural networks.
# <br>
# Johns Hopkins APL Technical Digest, 10, 262-266*.
# 
# <p style="text-align: right;"> Πηγή: https://archive.ics.uci.edu/ml/datasets/ionosphere </p>

# ## Προεπεξεργασία του Dataset

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

pd.options.display.max_columns = None #εμφάνισε όλες τις στήλες


# ### Μια πρώτη ματιά
# 
# Το dataframe περιέχει **35 στήλες**-attributes. Δύο για κάθε παλμό -όπως περιγράφηκε παραπάνω- άρα $2\cdot17=34$ στήλες καθώς και τη στήλη της κατηγορίοποίησης του κάθε στιγμιοτύπου. Ξεκινούμε την εξερεύνηση του dataset, τυπώνοντας μια στατιστική περίληψη των attributes:

# In[2]:


df = pd.read_csv('ionosphere_ds.csv')
#στατιστικά των 34 ποσοτικών μεταβλητων-στηλών
df.describe()


# In[3]:


#και η 35η στήλη της κατηγοριοποίησης
pd.DataFrame(df.loc[:,'label']).transpose()


# ### Περιττά Δεδομένα & ΝAs
# 
# Μελετώντας την παραπάνω στατιστική περίληψη των μεταβλητών, παρατηρούμε ότι η **δεύτερη στήλη** (όνομα μεταβλητης: `a02`) του dataframe (και μοναχά αυτή) έχει **μηδενική διακύμανση**, άρα είναι **αδιάφορη στήλη** και μπορούμε να την αφαιρέσουμε από το dataframe, στο πλαίσιο της μελέτης μας:

# In[297]:


#df.drop(columns = ['a02'], inplace = True)
#df.iloc[:3,:4]
#επιλέξαμε να αφαιρέσουμε τις στήλες με χαμηλή/μηδενική διακύμανση, στο grid search


# Επιπλέον, επιβεβαιώνουμε την πληροφορία από το description του dataset, ότι **δεν περιέχει missing values** (NAs):

# In[4]:


df.isna().any().any()


# ### Κωδικοποίηση των labels
# 
# To PyTorch, δέχεται ως είσοδο -στις loss functions- ακέραιους αριθμούς, ως δείκτες για την κατηγοριά (class) στην οποία ανήκει το κάθε στιγμιότυπο.
# Συνεπώς, θα χρειαστεί να αλλάξουμε το format των τιμών στη στήλη label του dataframe, από "g" (goob signal) ή "b" (bad signal) σε 1 και 0, αντίστοιχα.
# Επίσης, μετατρέπουμε τις παρατηρήσεις των μη διατάξιμων μεταβλητων - εδώ η λογική μεταβλητή στη στήλη `a01`- σε πραγματικούς αριθμούς.
# 
# Στην έξοδο του επόμενου κελιού, βλέπουμε τη μετασχηματισμένη στήλη των labels:

# In[5]:


df['label'] = df.label.astype('category')   

encoding = {'g': 1, 'b': 0}
df.label.replace(encoding, inplace = True)

df['a01'] = df.a01.astype('float64')

pd.DataFrame(df.loc[:,'label']).transpose()


# ### Κατανομή των labels
# 
# Όπως βλέπουμε, το dataset παρουσιάζει **ανισορροπία** μεταξύ των στιγμιοτύπων-σημάτων που έχουν χαρακτηριστεί ως g ή b (good ή bad)

# In[6]:


sns.countplot(x = 'label', data = df, palette = "Accent")


# In[7]:


df['label'].value_counts() #εμφανίσεις 0 και 1 στη στήλη label


# Οι συχνότητες εμφάνισης των 2 κλάσεων -καλά/κακά σήματα- είναι $\frac{225}{351}\approx0.64$ και $\frac{126}{351}\approx0.36$ αντίστοιχα.
# 
# Η κλάση "`good signal`" είναι $\frac{64}{36} = 1.7\bar{7}$ φορές συχνότερη από την κλάση "`bad signal`".
# 
# Συνεπώς, το dataset κρίνεται ως **μη ισορροπημένο**.

# ### Seperation & Split
# 
# Χωρίζουμε τις ανεξάρτητες από τις εξαρτημένες μεταβλητές (seperation). Η είσοδος Χ θα έχει **33 μεταβλητές** (34 - σταθερή μεταβλητή `a02`) και **351 παρατηρήσεις** και η έξοδος y θα είναι ένα διάνυσμα με τον ίδιο αριθμό παρατηρήσεων:

# In[8]:


X = df.values[:, :-1]
X.shape


# In[9]:


y = df.values[:, -1]
y.shape


# Υλοποιούμε το **διαχωρισμό** του dataset, σε **train** και **test** set (split), και κανονικοποιούμε τις παρατηρήσεις (scale), με τη βοήθεια των `train_test_split` και `StandardScaler` αντίστοιχα, από τη βιβλιοθήκη `sklearn`:

# In[2]:


from sklearn.model_selection import train_test_split


# In[10]:


# η παράμετρος random_state είναι ο seed για το shuffling.
# κάνει την έξοδο του αλγορίθμου αναπαράξιμη (reproducible output)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

len(x_test)


# ## Εποπτεία Διαχωρισμών

# Η οπτικοποίηση των δεδομένων του trainset (`x_train`) θα μας δώσει μια ιδέα για το πόσο σύνθετο θα είναι το πρόβλημα της ταξινόμησής τους.
# 
# Θα κάνουμε χρήση 2 αλγορίθμων απεικόνισης, οι οποίοι περιέχονται στη βιβλιοθήκη sklearn:

# >### t-SNE `(t-distributed stochastic neighbor embedding)`
# 
# 1. Προβολή ν-διάστατων στιγμιοτύπων στο κανονικό επίπεδο, σεβόμενοι τη **σχετική** τους απόσταση. 
# 2. Ελαχιστοποίηση Kullback-Leibler.
# 3. Non convex cost function, συνεπώς διαφορετικές απεικονίσεις σε κάθε επανάληψη. 
# 
# Τρέχουμε τον αλγόριθμο 2 φορες και επιβεβαιώνουμε πειραματικά:

# In[11]:


from sklearn.manifold import TSNE


# In[12]:


# πρώτο τρέξιμο t-SNE
x_embedded = TSNE(n_components = 2).fit_transform(x_train)

plt.scatter(x_embedded[:, 0], 
           x_embedded[:, 1], 
           color=['yellow' if label else 'purple' for label in y_train])
plt.show()

# δεύτερο τρέξιμο t-SNE
x_embedded = TSNE(n_components = 2).fit_transform(x_train)

plt.scatter(x_embedded[:, 0], 
           x_embedded[:, 1], 
           color=['yellow' if label else 'purple' for label in y_train])
plt.show()


# >### PCA `(Principal Component Analysis)`
# 
# 1. Επαναπροσδιορισμός συστήματος συντεταγμένων.
# 2. Aποτέλεσμα ενός γραμμικού συνδυασμού προερχόμενου από τις αρχικές μεταβλητές.
# 3. Eκπροσωπούνται σε ορθογώνιο άξονα.
# 4. Tα επικείμενα σημεία διατηρούν μια φθίνουσα σειρά στην τιμή της διακύμανσής τους.

# In[13]:


from sklearn.decomposition import PCA


# In[14]:


x_embedded = PCA(n_components = 2).fit_transform(x_train)

plt.scatter(x_embedded[:, 0], 
            x_embedded[:, 1], 
            color = ['red' if label else 'blue' for label in y_train])

plt.show()


# Από τις παραπάνω απεικονίσεις, προκύπτει ότι μελετάμε ένα σχετικά **απλό σύνολο**. Δοκιμάζοντας t-SNE απεικονίσεις, μπορούμε να πετύχουμε σχεδόν / ακόμη και γραμμικό διαχωρισμό.
# 
# Περιμένουμε υψηλή ακρίβεια από το μοντέλο μας. 

# ## Ταξινόμηση

# ### Baseline Classification
# 
# Θα ξεκινήσουμε τη μοντελοποίηση, εκτελώντας ορισμένες dummy στρατηγικές ταξινόμησης του dataset μας. Οι dummy classifiers ταξινομούν με κάποιους απλούς κανόνες-στρατηγικές τα δεδομένα. Χρησιμοποιούνται ως benchmarks για το scaling και ranking άλλων μοντέλων.
# 
# 
# Θα χρησιμοποιήσουμε τον `DummyClassifier`της sklearn για να εξετάσουμε την ακρίβεια όλων των διαθέσιμων στρατηγικών:

# In[22]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix 


# In[16]:


strategies = ['most_frequent', 'stratified', 'uniform', 'constant'] 
dummy_scores = [] 
con_matrices = []

for s in strategies: 
    
    if s =='constant': 
        dclf = DummyClassifier(strategy = s, constant = 1) 
    else: 
        dclf = DummyClassifier(strategy = s) 
        
    dclf.fit(x_train, y_train)

    dummy_scores.append(dclf.score(x_test, y_test)) 
    con_matrices.append(confusion_matrix(y_test, dclf.predict(x_test)))


# Ας δούμε τους πίνακες σύγχυσης των dummy στρατηγικών:

# In[31]:


fig, ax_lst = plt.subplots(1,4,figsize=(10,7))

for i,ax in zip(range(4), ax_lst.flat):
    hmap = sns.heatmap(
         con_matrices[i].T,
         cmap = 'Oranges',
         square = True,
         annot = True, 
         fmt = 'd',
         cbar = False,
         xticklabels = ['b','g'],
         yticklabels = ['b','g'])

    # Plot heatmap
    plt.sca(ax) # make the ax object the "current axes"
    plt.title(strategies[i], fontsize=16)
    plt.plot()
    
plt.show()


# Και ένα plot των scores τους:

# In[32]:


dummy_scores.sort()    
sns.set_theme(style = "whitegrid")

ax = sns.stripplot(x = strategies, y = dummy_scores, size = 10, palette = 'Oranges')
plt.xlabel('Strategy', fontsize=12)
plt.ylabel('Dummy Score', fontsize=12)
plt.show()


# ### Gaussian Naive Bayes
# 
# Ας δούμε πως ταξινομεί τα δεδομένα μας ο Gaussian Naive Bayes ταξινομητής:

# In[41]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


# #### Ένας απλός NB Ταξινομητής 
# 
# Για αρχή, κάνουμε μια **baseline ταξινόμηση**, με έναν απλό ταξινομητή, χωρίς επεξεργασία παραμέτρων:

# In[42]:


nbclf = GaussianNB()

nbclf.fit(x_train,y_train)
pred = nbclf.predict(x_test)


# In[43]:


target_names = ['good_signal', 'bad_signal']
print(classification_report(y_test, pred, target_names=target_names))

confusion = confusion_matrix(y_test, pred)

hmap = sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = True,
     xticklabels = ['g','b'],
     yticklabels = ['g','b'])

plt.title("Naive Bayesian", fontsize=16) 
plt.show()


# Παρατηρούμε ότι o απλός NB ταξινομητής έχει αρκετά υψηλή ακρίβεια.

# In[3]:


from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[45]:


pca = PCA()
selector = VarianceThreshold()
scaler = StandardScaler()
ros = RandomOverSampler()

pipe = Pipeline(steps =[('pca', pca),
                        #('scaler',scaler),
                        ('selector', selector), 
                        ('bayesian', nbclf)])


# Πειραματιστήκαμε με διάφορες αρχιτεκτονικές στο pipeline. To scaling, η μείωση των features και η αφαίρεση των στηλών με χαμηλή διακύμανση, δεν προέκυψε να επηρρεάζει τα scores του classifier, συγκρινόμενα με αυτά του baseline classification παραπάνω. Αποδίδουμε αυτό το αποτέλεσμα στη χαμηλή πολυπλοκότητα του dataset.

# #### Grid Search με NB  
# 
# Απεικονίζουμε γραφικά τη διακύμανση στις στήλες του dataset, για να μας βοηθήσει να αρχικοποιήσουμε τα φίλτρα χαμηλής διακύμανσης:

# In[46]:


plt.plot(x_train.var(axis = 0))
plt.ylabel('Variance')
plt.xlabel('Feature')
plt.show()


# Θα εξετάσουμε την επίδραση ορισμένων παραμέτρων στην επίδοση της ταξινόμησης με τον Naive Bayes ταξινομητή. Βελτιστοποιήσαμε το πλήθος των components, το φίλτρο για στήλες χαμηλής διακύμανσης και παραθέτουμε τα αποτελέσματα για τις μετρικές f1-micro και f-macro.
# 
# Σημειώνουμε πως επειδή πρόκειται για πρόβλημα δυαδικής ταξινόμησης, περιμένουμε και οι 2 μετρικές να μας δίνουν το ίδιο αποτέλεσμα, πράγμα που συμβαίνει:

# In[4]:


from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# In[51]:


vthreshold = [0, 0.1, 0.2, 0.3]
n_components = [20, 21, 22, 23, 24, 25, 26]
start_time = time.time()

gscv = GridSearchCV(pipe,
                 dict(selector__threshold=vthreshold,
                      pca__n_components=n_components),
                 cv = 10,
                 scoring = 'f1_micro',
                 n_jobs = -1
                )
gscv.fit(x_train,y_train)
preds = gscv.predict(x_test)


# In[52]:


print('Συνολικός χρόνος τρεξίματος:' ,time.time() - start_time)
print(classification_report(y_test,preds,target_names=target_names))

confusion = confusion_matrix(y_test, pred)

hmap = sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = True,
     xticklabels = ['g','b'],
     yticklabels = ['g','b'])

plt.title("Naive Bayesian", fontsize=16) 
plt.show()


# Ο παραπάνω, είναι o confusion matrix για βελτιστοποίηση του f1-micro score και από κάτω είναι οι παράμετροι που δίνουν τη μεγαλύτερη ακρίβεια.

# In[54]:


gscv.best_params_


# Επαναλαμβάνουμε, όπως προαναφέρθηκε, το grid search, βελτιστοποιώντας το f1-macro score.

# In[59]:


vthreshold = [0 , 0.1 , 0.2, 0.3]
n_components = [5,21,22,25,26,33]

gscv = GridSearchCV(pipe,
                 dict(selector__threshold=vthreshold, pca__n_components=n_components),
                 cv = 10,
                 scoring = 'f1_macro',
                 n_jobs = -1
                )
gscv.fit(x_train,y_train)
preds = gscv.predict(x_test)


# In[60]:


print(classification_report(y_test,preds,target_names=target_names))
confusion = confusion_matrix(y_test, pred)

hmap = sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = True,
     xticklabels = ['g','b'],
     yticklabels = ['g','b'])

plt.title("Naive Bayesian", fontsize=16) 
plt.show()


# ### K Nearest Neighbors
# 
# Τώρα θα δούμε πως ταξινομούν οι Κ Nearest Neighbors ταξινομητές το Ionosphere Dataset.
# 
# Για αρχή τρέχουμε έναν απλό KNN Classifier για baseline classification και στη συνέχεια βελτιστοποιούμε τις επιθυμητές παραμέτρους, μέσω grid search:

# In[61]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score


# In[62]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
preds = knn.predict(x_test)
print(classification_report(y_test,preds,target_names=target_names))


# Ας δούμε τι ακρίβεια πετυχαίνουμε, πειραματιζόμενοι με τις διαθέσιμες παραμέτρους του grid search:

# In[67]:


pca = PCA()
selector = VarianceThreshold(threshold=0.2)
scaler = StandardScaler()
ros = RandomOverSampler(sampling_strategy=1)

pipe = Pipeline(steps =[('pca',pca), 
                        #('scaler',scaler),
                        ('selector',selector), 
                        ('knn',knn)],
                        memory='tmp')

vthreshold = [0, 0.1, 0.2, 0.3]
n_components = [5,21,22,25,26,33]
neighbors = [1,2,3,4,5,6,7]

start_time = time.time()
gknn = GridSearchCV(
    pipe,
    dict(selector__threshold = vthreshold, 
         pca__n_components = n_components,
         knn__n_neighbors = neighbors),
    scoring='f1_micro',
    cv=10,
    n_jobs=-1,
    return_train_score = True
)

gknn.fit(x_train, y_train)
gknn.predict(x_test)

print('Συνολικός χρόνος τρεξίματος:' ,time.time() - start_time)
print(classification_report(y_test,pred))

#print(gknn.best_score_)


# In[68]:


print(gknn.best_params_)


# Παρατηρούμε ότι ο kNN classifier ταξινομεί αποδοτικότερα στους 2 κοντινότερους γείτονες. H baseline ταξινόμηση, απέδωσε καλύτερα από το grid search! Το dataset είναι αρκετά απλό. Συνεχίζουμε με το δεύτερο και μεγαλύτερο dataset.

# # Isolet Dataset
# 
# Λίγα λόγια για την προέλευση των δεδομένων:
# <div class="alert alert-block alert-success">
# Τα δεδομένα του Isolet dataset συλλέχθηκαν ως εξής: 150 συμμετέχοντες προέφεραν τους **26 φθόγγους** του λατινικού αλφαβήτου, 2 φορές τον καθένα. Συνεπώς **για κάθε συμμετέχοντα**, ηχογραφήθηκαν **52 παρατηρήσεις**. Τρεις παρατηρήσεις λέιπουν, πιθανά λόγω δυσκολίας στην ηχογράφηση. Το dataset επομένως περιέχει $150\cdot 52 - 3 = \boldsymbol{7797}$ **παρατηρήσεις**.
# <br><br>
# Η κάθε ηχογράφηση κωδικοποιήθηκε σε 617 πραγματικές μεταβλητές και κατηγοριοποιήθηκε σε μια νέα στήλη που περιέχει έναν κωδικό για κάθε φθόγγο. Άρα το dataset περιλαμβάνει **618 στήλες**.
# </div>
# <br>
# `Παραχωρήθηκε από:`
# 
# > *Department of Computer Science, Oregon State University, Corvallis*.
# 
# 
# 
# `Σχετικά Άρθρα:`
# 
# > 1. S Fanty, M., Cole, R. (1991).
# <br>
# *Spoken letter recognition. In Lippman, R. P., Moody, J., and Touretzky, D. S. (Eds).
# <br>
# Advances in Neural Information Processing Systems 3. San Mateo, CA: Morgan Kaufmann.*
# 
# >2. Dietterich, T. G., Bakiri, G. (1991)
# <br>
# *Error-correcting output codes: A general method for improving multiclass inductive learning programs.
# <br>
# Proceedings of the Ninth National Conference on Artificial Intelligence (AAAI-91), Anaheim, CA: AAAI Press.*
# 
# >3. Dietterich, T. G., Bakiri, G. (1994)
# <br>
# *Solving Multiclass Learning Problems via Error-Correcting Output Codes. *
# 
# <p style="text-align: right;"> Πηγή: https://archive.ics.uci.edu/ml/datasets/isolet </p>

# ## Προεπεξεργασία του Dataset

# ### Μια πρώτη ματιά
# 
# Το dataframe περιέχει **618 στήλες**-attributes. Ξεκινούμε την εξερεύνηση του dataset, τυπώνοντας μια στατιστική περίληψη των attributes:

# In[11]:


df = pd.read_csv('isolet_ds.csv')
#στατιστικά των 617 ποσοτικών μεταβλητων-στηλών
df.describe()


# In[12]:


#και η 618η στήλη της κατηγοριοποίησης
pd.DataFrame(df['label']).transpose()


# Αφαιρούμε τα single quotes, από τα ονόματα των κλάσεων (φθόγγων) στη στήλη `label`:

# In[4]:


# df['label'] = df['label'].str.replace('\'','')
# pd.DataFrame(df['label']).transpose()


# ### Περιττά Δεδομένα & ΝAs
# 
# Μελετώντας την παραπάνω στατιστική περίληψη των μεταβλητών, παρατηρούμε ότι δεν υπάρχει σταθερή στήλη (μας επιστρέφεται μια κενή λίστα):

# In[6]:


#df.loc[:, (df != df.iloc[0]).any()].count()
df.columns[df.nunique() < 2] #πόσες στήλες έχουν 1 τιμή; (σταθερές)


# Επιπλέον, επιβεβαιώνουμε την πληροφορία από το description του dataset, ότι **δεν περιέχει missing values** (NAs):

# In[7]:


df.isna().any().any()


# ### Κωδικοποίηση των labels
# 
# 

# In[13]:


df['label'] = df.label.astype('category')


# In[10]:


#df.loc['label'].value_counts()
df.select_dtypes(exclude = np.number) #πόσες μεταβ. δεν είναι αριθμητικές;


# ### Κατανομή των labels
# 
# Ας εξετάσουμε αν το dataset μας είναι ισορροπημένο:

# In[11]:


plt.figure(figsize=(12,4))
sns.countplot(x = 'label', data = df, palette = "Accent")


# In[12]:


pd.DataFrame(df['label'].value_counts()).transpose() # (α) στη στήλη label


# Καθαρά Ισορροπημένο το dataset μας.

# ### Seperation & Split
# 
# Χωρίζουμε τις ανεξάρτητες από τις εξαρτημένες μεταβλητές (seperation). Η είσοδος Χ θα έχει **617 μεταβλητές** και **7797 παρατηρήσεις** και η έξοδος y θα είναι ένα διάνυσμα με τον ίδιο αριθμό παρατηρήσεων:

# In[14]:


X = df.values[:, :-1]
y = df.values[:, -1]

print(X.shape,y.shape)


# Υλοποιούμε το **διαχωρισμό** του dataset, σε **train** και **test** set (split), και κανονικοποιούμε τις παρατηρήσεις (scale), με τη βοήθεια των `train_test_split` και `StandardScaler` αντίστοιχα, από τη βιβλιοθήκη `sklearn`:

# In[15]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 7)


# ## Ταξινόμηση

# ### Baseline Classification
# 
# Εκτελούμε ορισμένες dummy στρατηγικές ταξινόμησης του dataset μας. Οι dummy classifiers ταξινομούν με κάποιους απλούς κανόνες-στρατηγικές τα δεδομένα. Χρησιμοποιούνται ως benchmarks για το scaling και ranking άλλων μοντέλων.
# 
# 
# Θα χρησιμοποιήσουμε τον `DummyClassifier`της sklearn για να εξετάσουμε την ακρίβεια όλων των διαθέσιμων στρατηγικών:

# In[32]:


strategies = ['most_frequent', 'stratified', 'uniform', 'constant'] 

con_matrix = []
dummy_score = [] 

for s in strategies: 
    if s =='constant': 
        dclf = DummyClassifier(strategy = s, random_state = 0, constant = '3') # se poia stathera-gramma paei kalutera?
    else: 
        dclf = DummyClassifier(strategy = s, random_state = 0) 
    dclf.fit(x_train, y_train)
    score = dclf.score(x_test, y_test) 
    dummy_score.append(score) 
    con_matrix.append(confusion_matrix(y_test, dclf.predict(x_test)))    


# Απεικονίζουμε τους πίνακες σύγχυσης για όλες τις dummy στρατηγικές:

# In[34]:


fig, ax_lst = plt.subplots(4,1,figsize=(50,50))

for i,ax in zip(range(4), ax_lst.flat):
    hmap = sns.heatmap(
         con_matrix[i].T,
         cmap = 'Oranges',
         square = True,
         annot = True, 
         fmt = 'd',
         cbar = False)
         #xticklabels = ['b','g'],
         #yticklabels = ['b','g'])

    # Plot heatmap
    plt.sca(ax) # make the ax object the "current axes"
    plt.title(strategies[i], fontsize=16)
    
plt.show()


# In[35]:


sns.set_theme(style = "whitegrid")
ax = sns.stripplot(x = strategies, y = dummy_score, size = 9); 
ax.set(xlabel = 'Strategy', ylabel = 'Dummy Score') 
plt.show() 


# ### Gaussian Naive Bayes
# 
# Ας δούμε πως ταξινομεί τα δεδομένα μας ο NB:

# In[36]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import time


# #### Ένας απλός ΝΒ ταξινομητής

# In[37]:


nbclf = GaussianNB()

start_time = time.time()
model = nbclf.fit(x_train,y_train)
pred = model.predict(x_test)


# In[38]:


print('Συνολικος χρονος τρεξιματος' ,time.time() - start_time)

print(classification_report(y_test,pred))

fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for kNN:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# #### Grid Search με ΝΒ
# 
# Υλοποιούμε ένα pipeline και στη συνέχεια ταξινομούμε μόνο με αυτό και ύστερα με Grid Search για f1-micro και f2-macro βελτιστοποίηση. Παραθέτουμε classification reports και confusion matrices για κάθε πείραμα.

# In[46]:


selector = VarianceThreshold(threshold=0)
pca = PCA(n_components=150)
scaler = StandardScaler()

pipe = Pipeline(steps =[('pca',pca),
                        ('scaler',scaler), 
                        ('selector',selector),
                        ('bayesian',nbclf)],memory='tmp')

pipe.fit(x_train,y_train)
pred = pipe.predict(x_test)


# In[48]:


print(classification_report(y_test,pred))
fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for NB:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# In[50]:


vthreshold = [0.1, 0.2, 0.3]
n_components = [147,148,149,150,151]

start_time = time.time()

grid = GridSearchCV(
    pipe,
    dict(selector__threshold=vthreshold, 
         pca__n_components=n_components,),
    scoring = 'f1_macro',
    cv = 5,
    n_jobs = -1,
)

grid.fit(x_train, y_train)
pred = grid.predict(x_test)


# In[51]:


print('Συνολικος χρονος τρεξιματος:' ,time.time() - start_time)
print(classification_report(y_test,pred))
fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for NB:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# ### K Nearest Neighbors
# 
# 
# Τώρα θα δούμε πως ταξινομούν οι Κ Nearest Neighbors ταξινομητές.
# 
# #### Ένας απλός ΚΝΝ ταξινομητής
# Ξεκινάμε με έναν απλό KNN ταξινομητή και συνεχίζουμε, βελτιστοποιώντας τις παραμέτρους μέσω grid search:

# In[17]:


from sklearn.neighbors import KNeighborsClassifier 


# In[19]:


nclf = KNeighborsClassifier() 
nclf.fit(x_train,y_train)
pred = nclf.predict(x_test)


# In[23]:


print(classification_report(y_test,pred))
fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for KNN:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# #### Grid Search με KNN

# In[64]:


selector = VarianceThreshold()
pca = PCA()

pipe = Pipeline(steps =[ ('pca',pca), ('selector',selector), ('knn',nclf)] , memory='tmp')

vthreshold = [0.3, 0.35, 0.4]
n_components = [40,60,80]
n_neighbors = [13,14,15]

start_time = time.time()
knn = GridSearchCV(
    pipe,
    dict(selector__threshold=vthreshold,pca__n_components=n_components, knn__n_neighbors=n_neighbors),
    scoring='f1_micro',
    cv=5,
    n_jobs=-1
)

knn.fit(x_train, y_train)
pred = knn.predict(x_test)
print('Συνολικός χρόνος τρεξίματος:' ,time.time() - start_time)


# In[65]:


print(classification_report(y_test,pred))

fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for kNN:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# In[66]:


print(knn.best_params_)


# Παρατηρούμε ότι ο kNN classifier ταξινομεί αποδοτικότερα στους  κοντινότερους γείτονες.

# ### SVM
# 
# Θα ταξινομήσουμε το Isolet Dataset με χρήση του SVM Classifier. Υλοποιούμε 2 grid search, βελτιστοποιώντας τις f1-micro και f1-macro μετρικές αντίστοιχα:

# In[33]:


from sklearn.svm import SVC
from sklearn.utils import shuffle


# In[36]:


sdata, starget = shuffle(X, y, random_state=64327)
samples = 3000

data = sdata[0:samples-1,:]
target = starget[0:samples-1]

x_train, x_test, y_train, y_test = train_test_split(data, 
                                                    target, 
                                                    test_size = 0.3, 
                                                    random_state=74)
selector = VarianceThreshold()
pca = PCA()
svc = SVC(class_weight='balanced')

pipe = Pipeline(steps =[ ('pca',pca), ('selector',selector), ('svc',svc)],memory='tmp')

param_grid = {
    'pca__n_components': [265, 270, 275],
    'svc__kernel':['linear','poly','sigmoid','rbf'],
    #'svc__kernel':['rbf','linear'],
    'svc__C':[1,2,3,4],
    'svc__gamma': [0.0045,0.005,0.0055]
}

start_time = time.time()

grid = GridSearchCV(
    pipe,
    param_grid,
    scoring='f1_micro',
    cv=5,
    n_jobs=-1
    )

grid.fit(x_train, y_train)
pred = grid.predict(x_test)
print('Συνολικός χρόνος τρεξίματος:' ,time.time() - start_time)


# In[38]:


print(classification_report(y_test,pred))

fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for kNN:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# In[39]:


print(grid.best_params_)


# In[41]:


sdata, starget = shuffle(X, y, random_state=64327)
samples = 3000

data = sdata[0:samples-1,:]
target = starget[0:samples-1]

x_train, x_test, y_train, y_test = train_test_split(data, 
                                                    target, 
                                                    test_size = 0.3, 
                                                    random_state=74)

selector = VarianceThreshold()
pca = PCA()
svc = SVC(class_weight='balanced')

pipe = Pipeline(steps =[ ('pca',pca), ('selector',selector), ('svc',svc)],memory='tmp')

param_grid = {
    'pca__n_components': [280, 290, 300, 310, 320],
    #'svc__kernel':['linear','poly','sigmoid','rbf'],
    'svc__kernel':['rbf'],
    'svc__C':[6,7,8,9],
    'svc__gamma': [0.006, 0.0065, 0.007]
}

start_time = time.time()

grid = GridSearchCV(
    pipe,
    param_grid,
    scoring='f1_macro',
    cv=5,
    n_jobs=-1
    )

grid.fit(x_train, y_train)
pred = grid.predict(x_test)
print('Συνολικός χρόνος τρεξίματος:' ,time.time() - start_time)


# In[43]:


print(classification_report(y_test,pred))

fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for kNN:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# ### MLP
# 
# Τέλος, ταξινομούμε το dataset μας, με τη χρήση MLP (Multi Layer Perceptron) ταξινομητή. Βελτιστοποιούμε τις παραμερους που φαίνονται στη λίστα "param_grid" παρακάτω, μέσω grid search.

# In[25]:


from sklearn.neural_network import MLPClassifier


# In[27]:


mlp = MLPClassifier()

param_grid = {
    'hidden_layer_sizes': [(5,) , (15,) , (20,) , (50,) , (100,)], # tuple default (100,).The ith element represents the number of neurons in the ith hidden layer.
    'activation': ['identity', 'logistic', 'tanh', 'relu'], # default ‘relu’
    'solver': ['lbfgs','sgd','adam'], # default ‘adam’
    'learning_rate': ['constant', 'invscaling','adaptive'] # Only used when solver='sgd'.
    #'max_iter': [100,150,200,250],
    #'alpha': [0.0001, 0.002, 0.05]   
}

start_time = time.time()

clf = GridSearchCV(mlp,param_grid,n_jobs=-1,cv=5,scoring='f1_micro')

clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print('Συνολικός χρόνος εκτέλεσης:', time.time()-start_time)


# In[28]:


print(classification_report(y_test,pred))

fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for MLP:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# In[29]:


print(clf.best_params_) 


# Επαναλαμβάνουμε για f1-macro score:

# In[31]:


mlp = MLPClassifier()

param_grid = {
    'hidden_layer_sizes': [(5,) , (15,) , (20,) , (50,) , (100,)], # tuple default (100,).The ith element represents the number of neurons in the ith hidden layer.
    'activation': ['identity', 'logistic', 'tanh', 'relu'], # default ‘relu’
    'solver': ['lbfgs','sgd','adam'], # default ‘adam’
    'learning_rate': ['constant', 'invscaling','adaptive'] # Only used when solver='sgd'.
    #'max_iter': [100,150,200,250],
    #'alpha': [0.0001, 0.002, 0.05] 
}

start_time = time.time()

clf = GridSearchCV(mlp,param_grid,n_jobs=-1,cv=5,scoring='f1_macro')

clf.fit(x_train,y_train)
clf.predict(x_test)
print('Συνολικός χρόνος εκτέλεσης:', time.time()-start_time)


# In[32]:


print(classification_report(y_test,pred))

fig, ax_lst = plt.subplots(1,1,figsize=(12,12))
confusion = confusion_matrix(y_test, pred)
print('Confusion matrix for MLP:')
sns.heatmap(
     confusion.T,
     cmap = 'Oranges',
     square = True,
     annot = True, 
     fmt = 'd',
     cbar = False)


# > **Τέλος Εργασίας**

# Για τον κώδικά μας, χρησιμοποιήσαμε τη `standard` ονοματοδοσία (ακόμη και των μεταβλητών όπου υπάρχει τέτοια) που συναντήσαμε στα πλέον δημοφιλή -σχετικά με το αντικείμενο- sites/forums στο διαδίκτυο (https://www.geeksforgeeks.org/ ,https://stackoverflow.com/, https://www.kaggle.com/, https://towardsdatascience.com/, https://www.w3schools.com/ κα) όσο και στο documentation των βιβλιοθηκών που χρησιμοποιήσαμε (numpy, pandas, seaborn, pyplot, sklearn, torch κλπ).
# 
# Στο πλαίσιο των απαιτήσεων της εργασίας παραλείπουμε την αναλυτική αναφορά σε πηγές.
