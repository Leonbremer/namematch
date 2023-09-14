# Readme - Fuzzy name matching

## The algorithm

This project proposes a fuzzy name matching algorithm that uses existing computational methods. The algorithm takes four steps:
1. cleaning of character strings;
1. similarity scoring;
1. a decision rule; and
1. disambiguation of the resulting matching links.

Step 1: For the cleaning several functions are created. Strings are normalized, but also legal names like incorporated are recognized and either abbreviated or deleted.

Step 2: Similarity scoring is achieved by vectorizing the firm names using the TF-IDF method and sequentially calculating the angles between each vectorized firm name across databases using the Cosine Similarity. A cutoff on this angle is chosen to select a limited set of similar firm name matches, named candidates.

Step 3: Not all candidates are correct matches. Therefore a random sample of candidates is manually labelled. A probit regression is used to fit a model explaining true match status with name and link characteristics. A part of the labelled sample is used to create performance indicators. The cutoff on the predicted probability ($\bar{p}$) of a true match is determined by the maximum F1 score. The unlabelled set of candidates then gets their true matching status predicted using the fitted probit model and the determined $\hat{p}$.

Step 4: The resulting linkages suffer from multiplicity, meaning that applicants can link to multiple firms. This is for most follow-up applications not desirable. This last step provides three solutions for disambiguation. First the applicant's link with the highest $\hat{p}$ is chosen, leaving each applicant linked to one single firm. Second, a community detection algorithm determines which firms belong to each other based on their co-occurences in links with patent applicants. This creates firm groups to which applicants then belong. If applicants are linked to multiple groups they are assigned to the group with most firms. Third, the largest consolidated firm in the firm group can be taken to represent the entire firm group. Applicants are then uniquely linked to one consolidated firm.

## The application

The algorithm is applied to matching the Amadeus Financials database and the Amadeus Subsidiaries database, both containing data from financial statements, to PATSTAT, a worlwide patent database. The algorithm vastly outperformes a simple exact name match. The number of matched firms more than doubles. The number of matched applicants increases with more than 400%. And 18.1% of patent applications since 1950 have at least one applicant that can be linked to a firm in the Amadeus data, compared to 2.6%.

## The code

The code is a work in progress. Unfortunately I cannot share the data of the application. Also, the module parameters.py contains personal paths to files. When interested in replication one should create their own file with path information.

## Accompanying article and referencing

This work is also formalized in an academic article. It will soon be available as a Tinbergen Institute Discussion paper on the [TI website](https://tinbergen.nl/discussion-papers). When referencing my work, please refer to this article.
