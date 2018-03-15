# Phon_Phon
In Jeff Mielke's 2008 book *The Emergence of Distinctive Features*, it is argued
that SPE (*Sound Patterns of English*) style features are emergent rather than 
innate. In previous literature the grouping of sounds undergoing or triggering 
a phonological process is called a natural class. This matches Chomsky and Halle's 
notion that phonological processes target classes of sounds defined by an 
innate set of features. 

Since SPE's publication, however, there have been many examples of phonological 
processes that target a class of sounds that are not readily explained by 
a reduced set of these features. Mielke set out to explain these data. His conclusion
was that we have phonologically active classes of sounds, that are based 
on a language specific set of features learned by a child. However, no explict learning
algorithm is given by him that can derive such features. 

This project is part of an attempt to learn phonologically active classes from 
language data. I focus here on English. In particular, I want to explore
the classes that condition the allophones of the plural inflectional
morphology. The learning of a useful feature set from acoustic data is a complex
task, so we begin with a smaller exploration. I take as given that a child 
has learned a set of features similar to those given in Zsiga (2013) *The Sounds
of Language*. I then want to see if the features commonly used in the SPE style 
rules that derive the English morphology are indeed the best features.

To do this, I use a Logisitic Regression classifier, with the set of features for 
each sound as input and the morpheme as output. The data is taken from the 
English data from Universal Dependencies. I, then try a variety of statistical 
methods to determine which features contribute the most to the determination of the
output class. With these classes learned, I then create methods for seeing if we can 
generate associated morphemes for new words (held out data) and if ordering of 
presented data effects the feature set learned. 


## Usage

First clone the repo:

```
git clone https://github.com/forrestdavis/Phon_Phon.git
```

Then navigate to the scripts directory:

```
cd scripts/
```

Within scripts you have two programs, model.py and stats.py. In model.py, 
there are methods for loading different types of data (only stem final or
full stem), for training a model, for making predictions on data, and
for returning statistics about feature importance in a model. Running 
model.py will give you a good example of the types of information 
returned. Within the script stats.py, there is a method for determing
feature importance using sklearn ExtraTreesClassifer. 

In the directory tools there are a variety of python scripts for formating 
the data, such as padding to the maximum word length. 
