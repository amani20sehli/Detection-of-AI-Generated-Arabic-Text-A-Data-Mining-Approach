import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


# Load data
def load_and_compute(path='data/processed/train.csv'):
    df = pd.read_csv(path)

    df['word_count'] = df['text'].str.split().str.len()
    df['avg_word_len'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['sentence_count'] = df['text'].str.count(r'[.!?؟]') + 1
    df['ttr'] = df['text'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()) if str(x).split() else 0)

    return df


# Plot distributions
def plot_distributions(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    stats = ['word_count', 'avg_word_len', 'sentence_count', 'ttr']
    titles = ['Word Count', 'Avg Word Length', 'Sentence Count', 'Type-Token Ratio']

    for idx, (stat, title) in enumerate(zip(stats, titles)):
        ax = axes[idx // 2, idx % 2]
        df[df['label'] == 0][stat].hist(bins=30, alpha=0.6, label='Human', ax=ax, color='blue')
        df[df['label'] == 1][stat].hist(bins=30, alpha=0.6, label='AI', ax=ax, color='red')
        ax.set_xlabel(title)
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.savefig('reports/figures/distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


# Plot features
def plot_features(df):
    features = ['multiple_elongations', 'periods', 'verbs', 'dual_words', 'entity_diversity']

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        human_data = df[df['label'] == 0][feat]
        ai_data = df[df['label'] == 1][feat]
        ax.boxplot([human_data, ai_data], labels=['Human', 'AI'])
        ax.set_title(feat.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('reports/figures/features.png', dpi=300, bbox_inches='tight')
    plt.close()


# Word clouds
def plot_wordclouds(df):
    try:
        from wordcloud import WordCloud

        human_text = ' '.join(df[df['label'] == 0]['text'].astype(str))
        ai_text = ' '.join(df[df['label'] == 1]['text'].astype(str))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        wc_h = WordCloud(width=700, height=400, background_color='white').generate(human_text)
        axes[0].imshow(wc_h, interpolation='bilinear')
        axes[0].set_title('Human Texts', fontsize=14)
        axes[0].axis('off')

        wc_a = WordCloud(width=700, height=400, background_color='white').generate(ai_text)
        axes[1].imshow(wc_a, interpolation='bilinear')
        axes[1].set_title('AI Texts', fontsize=14)
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig('reports/figures/wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()

    except ImportError:
        pass


# N-grams
def plot_ngrams(df, n=2, top=15):
    def get_ngrams(text, n):
        words = str(text).split()
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    human_ng = []
    ai_ng = []

    for text in df[df['label'] == 0]['text']:
        human_ng.extend(get_ngrams(text, n))

    for text in df[df['label'] == 1]['text']:
        ai_ng.extend(get_ngrams(text, n))

    human_top = Counter(human_ng).most_common(top)
    ai_top = Counter(ai_ng).most_common(top)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if human_top:
        words, counts = zip(*human_top)
        axes[0].barh(range(len(words)), counts, color='blue', alpha=0.7)
        axes[0].set_yticks(range(len(words)))
        axes[0].set_yticklabels(words, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_title(f'Top {n}-grams (Human)')

    if ai_top:
        words, counts = zip(*ai_top)
        axes[1].barh(range(len(words)), counts, color='red', alpha=0.7)
        axes[1].set_yticks(range(len(words)))
        axes[1].set_yticklabels(words, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_title(f'Top {n}-grams (AI)')

    plt.tight_layout()
    plt.savefig('reports/figures/ngrams.png', dpi=300, bbox_inches='tight')
    plt.close()


# Save summary
def save_summary(df):
    summary = f"Train: {len(df)} samples\n\n"

    for stat in ['word_count', 'avg_word_len', 'sentence_count', 'ttr']:
        h = df[df['label'] == 0][stat].mean()
        a = df[df['label'] == 1][stat].mean()
        summary += f"{stat}: Human={h} AI={a}\n"

    for feat in ['multiple_elongations', 'periods', 'verbs', 'dual_words', 'entity_diversity']:
        h = df[df['label'] == 0][feat].mean()
        a = df[df['label'] == 1][feat].mean()
        summary += f"{feat}: Human={h} AI={a}\n"

    with open('reports/phase2_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)

    print(summary)


def main():
    df = load_and_compute()
    plot_distributions(df)
    plot_features(df)
    plot_wordclouds(df)
    plot_ngrams(df)
    save_summary(df)


if __name__ == "__main__":
    main()
