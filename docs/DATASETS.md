# Dataset sources, processing and usage restrictions

## MovieLens 100K and 1M

- Official index: https://grouplens.org/datasets/movielens/
- 100K archive: https://files.grouplens.org/datasets/movielens/ml-100k.zip
- 1M archive: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Format: MovieLens 100K `u.data` (tab separated); MovieLens 1M `ratings.dat` (`::` separated).

GroupLens asks users to review the README/terms distributed with each dataset and notes that public redistribution is generally not permitted. Do not commit the raw archives or processed copies to a public repository without checking the applicable terms. Cite the MovieLens dataset paper: Harper and Konstan, *The MovieLens Datasets: History and Context*, ACM Transactions on Interactive Intelligent Systems, 2015.

## Amazon Movies and TV

- Dataset information and terms: https://mcauleylab.ucsd.edu/public_datasets/data/amazon/datasets.html
- Historical 5-core ratings/reviews file: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz
- Format: one JSON object per line; the fields used here are `reviewerID`, `asin`, `overall`, and `unixReviewTime`.

The UCSD page states that the data provider does not own the underlying Amazon data and therefore cannot offer a data license; the page requests research use and provides per-category data. Check the source page and current terms before use or redistribution. Cite McAuley et al. (2015), as applicable to the downloaded Amazon review release.

## Netflix Prize

The manuscript describes a sampled Netflix Prize subset with 10,000 users, 15,000 items and 482,100 interactions after preprocessing. It does not provide the exact source-file list, deterministic sampling rule, or subset checksum. Obtain the Netflix Prize data through an authorized source and follow its applicable terms. The parser supports the classic `combined_data_*.txt` format. The optional `--sample-netflix` command in this repository is a transparent, deterministic substitute and is **not** guaranteed to reproduce the paper's unpublished subset.

## Common processing policy implemented here

1. Keep ratings greater than or equal to 4.0 as implicit positive feedback.
2. Drop duplicate `(user_id, item_id)` pairs; as an explicit implementation choice not fixed by the paper, the row with the latest timestamp is retained (stable sort breaks timestamp ties deterministically).
3. Iteratively remove users and items with fewer than five remaining interactions. The manuscript states the minimum-count rule but does not explicitly state whether filtering is iterative; iterative k-core is the implementation choice here.
4. Sort each user's interactions by timestamp (stable tie-break on item ID).
5. Last interaction is test; second-to-last is validation; earlier interactions are train.
6. Remap user/item identifiers to contiguous integers and save arrays in NPZ format.

The actual output counts are always printed and saved as JSON beside the processed dataset. The manuscript's tabulated counts are available in `paper_reference/dataset_statistics.csv` for auditing.
