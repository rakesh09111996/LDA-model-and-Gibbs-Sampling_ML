Task 1: Gibbs Sampling
In this portion, Our task is to implement the collapsed Gibbs sampler for LDA. In the case of LDA, the output represents a sample of the (hidden) topic variables for each word. Recall that in LDA we sample the hidden topic variables associated with words in the text. This sample of topic variables can be used to calculate topic representations per document. Algorithm 1 describes one possible implementation of the collapsed Gibbs sampler. For this project, we fix the number of iterations to run the sampler at Niters = 500. The Dirichlet parameter for the topic distribution is α1 where 1 is a vector of ones with K entries (K is the number of topics), and α = 5 K . The Dirichlet parameter for the word distribution is β1 where 1 is a vector of ones with V entries (V is the size of the vocabulary), and β = 0.01. We suggest testing your implementation first on the artificial dataset with K = 2 because you know what to expect and the run time is shorter. Once you have verified that your implementation works correctly, run your sampler with K = 20 on the 20 newsgroups dataset. After the sampler has finished running, output the 5 most frequent words of each topic into a CSV file, topicwords.csv, where each row represents a topic. Include these results in both your report and submission. In your report discuss the results obtained (i.e., the topics). Do the topics obtained make sense for the dataset? Finally, you will need the topic representations for the next part. For a document doc, this will be a vector of K values, one for each topic, where the kth value is given by (Cd(doc,k)+α) /(Kα+∑l Cd(doc,l))and Cd is output from the sampler.

Task 2: Classification
In this portion we will evaluate the dimensionality reduction accomplished by LDA in its ability to support document classification and compare it to the bag of words representation. The first step is to prepare the data files for the two representations. The first is given by the topic representation of the previous section, where each document is represented by a feature vector of length K. The second representation is the “bag-of-words” representation. This representation has a feature for each word in the vocabulary and the value of this feature is the number of occurrences of the corresponding word in the document. For the evaluation we will reuse the logistic regression implementation from project 3, in  particular, our implementation of Newton’s method for this problem. We use the value α = 0.01 for the regularization parameter of logistic regression in this part.
The task is to generate learning curves in the same way you did there: Step 1) Set aside 1/3 of the total data (randomly selected) to use as a test set. Step 2) Record test set classification error as a function of increasing training set size (with each training set randomly selected from the other 2/3 of the total data). Repeat Steps 1 & 2 a total of 30 times to generate learning curves with error bars (i.e., ±1σ). Performance is defined as classification accuracy on the test set.
Plot the learning curve performance of the logistic regression algorithm (with error bars) on the two representations. Then discuss your observations on the results obtained, as well as the runtime of the algorithms.

Algorithm 1 
Collapsed Gibbs sampler for LDA
Require: Number of topics K, Dirichlet parameter for topic distribution α, Dirichlet parameter
for word distribution β, number of iterations to run sampler Niters, array of word indices w(n),
array of document indices d(n), and array of initial topic indices z(n), where n = 1 . . . Nwords
and Nwords is the total amount of words in the corpus.
1: Generate a random permutation π(n) of the set {1, 2, . . . , Nwords}
2: Initialize a D×K matrix of topic counts per document Cd, where D is the number of documents
3: Initialize a K × V matrix of word counts per topic Ct, where V is the number of words in the
vocabulary
4: Initialize a 1 × K array of probabilities P (to zero)
5: for i = 1 to Niters do
6: for n = 1 to Nwords do
7: word ← w(π(n))
8: topic ← z(π(n))
9: doc ← d(π(n))
10: Cd(doc, topic) ← Cd(doc, topic) − 1
11: Ct(topic, word) ← Ct(topic, word) − 1
12: for k = 1 to K do
13: P (k) = ((Ct(k,word)+β)/V β+∑j Ct(k,j))*((Cd(doc,k)+α)/(Kα+∑l Cd(doc,l))
14: end for
15: P ← normalize P
16: topic ← sample from P
17: z(π(n)) ← topic
18: Cd(doc, topic) ← Cd(doc, topic) + 1
19: Ct(topic, word) ← Ct(topic, word) + 1
20: end for
21: end for
22: return {z(n)}, Cd, Ct


