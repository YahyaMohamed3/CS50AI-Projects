import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = {}
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        try:
            with open(os.path.join(directory, filename)) as f:
                contents = f.read()
                links = re.findall(r'<a\s+(?:[^>]*?)href="([^"]*)"', contents)
                pages[filename] = set(links) - {filename}
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    for filename in pages:
        pages[filename] = {link for link in pages[filename] if link in pages}

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    total_pages = len(corpus)
    prob_dist = {}

    if corpus[page]:
        link_prob = damping_factor / len(corpus[page])
        for p in corpus:
            prob_dist[p] = (1 - damping_factor) / total_pages
            if p in corpus[page]:
                prob_dist[p] += link_prob
    else:
        for p in corpus:
            prob_dist[p] = 1 / total_pages

    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_ranks = {page: 0 for page in corpus}
    current_page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_ranks[current_page] += 1
        transition_probs = transition_model(corpus, current_page, damping_factor)
        next_page = random.choices(
            population=list(transition_probs.keys()),
            weights=list(transition_probs.values()),
            k=1
        )[0]
        current_page = next_page

    for page in page_ranks:
        page_ranks[page] /= n

    return page_ranks


def iterate_pagerank(corpus, damping_factor, epsilon=0.001):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    page_ranks = {page: 1 / N for page in corpus}
    converged = False

    while not converged:
        new_ranks = {}
        for page in corpus:
            rank_sum = 0
            for potential_linker in corpus:
                if page in corpus[potential_linker]:
                    rank_sum += page_ranks[potential_linker] / len(corpus[potential_linker])
                elif len(corpus[potential_linker]) == 0:
                    rank_sum += page_ranks[potential_linker] / N
            new_ranks[page] = (1 - damping_factor) / N + damping_factor * rank_sum

        converged = all(abs(new_ranks[page] - page_ranks[page]) < epsilon for page in corpus)
        page_ranks = new_ranks

    return page_ranks


if __name__ == "__main__":
    main()
