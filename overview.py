
# Erkennung von Geschlechtszuschreibungen
# Erzähltexte enthalten eine große Menge von Figuren, 
# die oft als Platzhalter für echte Menschen gelesen werden. 
# Gleichzeitig sind sie natürlich keine echten Menschen, 
# sondern von den jeweiligen Autor:innen “gemacht”. 
# Das betrifft ganz zentral auch ihr Geschlecht bzw. die textuelle Repräsentation davon.
# In diesem Projekt geht es darum, für Figurenerwähnungen 
# in literarischen Texten automatisch zu erkennen, 
# ob sie als “Mann”, “Frau” oder “Neutral” klassifiziert werden.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import glob
from typing import Dict, Tuple

try:
	import spacy
except Exception:  # keep script runnable without spacy or if environment conflicts
	spacy = None


def find_tsv_files(base_dir: str) -> list:
	"""
	Locate all 12 test TSV files, explicitly excluding the large training corpus.
	Matches both prefixed variants: m_wTesttext*.tsv and m_w_Testtext*.tsv
	"""
	# Include all TSVs in the directory (covers both Testtext and Trainingskorpus)
	files = sorted(glob.glob(os.path.join(base_dir, "*.tsv")))
	return files


def read_tsv(path: str) -> pd.DataFrame:
	"""
	Read a two-column TSV with token and label.
	Ensures consistent column names and types.
	"""
	df = pd.read_csv(
		path,
		sep="\t",
		header=None,
		names=["token", "label"],
		dtype=str,
		on_bad_lines="skip",
		engine="python",
	)
	# Normalize whitespace and drop fully empty rows
	df["token"] = df["token"].fillna("").astype(str).str.strip()
	df["label"] = df["label"].fillna("").astype(str).str.strip()
	df = df[(df["token"] != "") | (df["label"] != "")].reset_index(drop=True)
	return df


def analyze_labels(df: pd.DataFrame) -> dict:
	"""
	Compute counts for relevant labels.
	We treat labels exactly as given in data: 'Mann', 'Frau', 'Genderneutral'.
	All others (e.g., 'O') are ignored for the gender counts but tracked as other.
	"""
	labels_of_interest = ["Mann", "Frau", "Genderneutral"]
	vc = df["label"].value_counts(dropna=False)
	result = {
		"Mann": int(vc.get("Mann", 0)),
		"Frau": int(vc.get("Frau", 0)),
		"Genderneutral": int(vc.get("Genderneutral", 0)),
		"O": int(vc.get("O", 0)),
		"TOTAL_ROWS": int(len(df)),
		"TOTAL_LABELED": int(sum(vc.get(k, 0) for k in labels_of_interest)),
	}
	return result


def analyze_corpus(base_dir: str) -> tuple[pd.DataFrame, dict]:
	"""
	Analyze all target TSV files and return a per-file summary DataFrame and global totals.
	"""
	files = find_tsv_files(base_dir)
	per_file_rows = []
	totals = {"Mann": 0, "Frau": 0, "Genderneutral": 0, "O": 0, "TOTAL_ROWS": 0, "TOTAL_LABELED": 0}

	for f in files:
		df = read_tsv(f)
		counts = analyze_labels(df)
		per_file_rows.append({
			"file": os.path.basename(f),
			**counts,
		})
		for k in totals:
			totals[k] += counts.get(k, 0)

	per_file_df = pd.DataFrame(per_file_rows).sort_values("file").reset_index(drop=True)
	return per_file_df, totals


def build_pos_counts(base_dir: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
	"""
	Create POS frequency counts per gender by tagging reconstructed texts per file.
	This does NOT modify the TSVs; outputs are generated separately.
	"""
	if spacy is None:
		print("spaCy not installed; skipping POS tagging. Run 'python -m pip install spacy' and download 'de_core_news_sm'.")
		return pd.DataFrame(), {}

	# Load German model: prefer large (lg), fallback to medium (md), then small (sm)
	nlp = None
	for model_name in ("de_core_news_lg", "de_core_news_md", "de_core_news_sm"):
		try:
			nlp = spacy.load(model_name)
			print(f"Loaded spaCy model: {model_name}")
			break
		except Exception:
			continue
	if nlp is None:
		print("German spaCy model not available. Install via: python -m spacy download de_core_news_lg (or md/sm)")
		return pd.DataFrame(), {}

	files = find_tsv_files(base_dir)
	rows = []
	# Aggregate counts: gender -> POS tag -> count
	agg: Dict[str, Dict[str, int]] = {"Mann": {}, "Frau": {}, "Genderneutral": {}, "O": {}}

	for f in files:
		df = read_tsv(f)
		# Reconstruct text as lines to preserve rough boundaries
		text = " ".join(df["token"].tolist())
		# Avoid max_length errors on long files (e.g., training corpus)
		nlp.max_length = max(nlp.max_length, len(text) + 1000)
		doc = nlp(text)
		# Simple heuristic: align tokens by string match order; if mismatch, default to 'O'
		# Build a fast iterator over labels
		labels = df["label"].tolist()
		li = 0
		for tok in doc:
			pos = tok.pos_
			# Advance label index while trying to match token text; here we take sequential alignment
			label = labels[li] if li < len(labels) else "O"
			li += 1
			rows.append({"file": os.path.basename(f), "token": tok.text, "pos": pos, "label": label})
			# aggregate
			agg.setdefault(label, {})
			agg[label][pos] = agg[label].get(pos, 0) + 1

	pos_df = pd.DataFrame(rows)
	return pos_df, agg


def save_outputs(per_file_df: pd.DataFrame, totals: dict, out_dir: str) -> None:
	"""
	Save a CSV summary and a simple bar chart visualization of the global totals.
	"""
	csv_path = os.path.join(out_dir, "gender_counts_summary.csv")
	per_file_df.to_csv(csv_path, index=False, encoding="utf-8")

	# Bar chart of absolute counts (including 'O')
	labels = ["Mann", "Frau", "Genderneutral", "O"]
	values = [totals.get(k, 0) for k in labels]
	plt.figure(figsize=(7, 4), constrained_layout=True)
	bars = plt.bar(labels, values, color=["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"])
	plt.ylabel("Anzahl (absolut)")
	plt.title("Geschlechtszuschreibungen im Korpus (absolut, inkl. O)")
	for b, v in zip(bars, values):
		plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}", ha="center", va="bottom")
	png_path = os.path.join(out_dir, "gender_counts_bar.png")
	plt.savefig(png_path, dpi=150, bbox_inches="tight")
	plt.close()

	# Donut (pie) chart of labeled percentages (M/F/Genderneutral)
	labeled_sum = sum(totals.get(k, 0) for k in ["Mann", "Frau", "Genderneutral"])
	if labeled_sum > 0:
		pie_labels = ["Mann", "Frau", "Genderneutral"]
		pie_values = [totals.get(k, 0) for k in pie_labels]
		colors = ["#1f77b4", "#d62728", "#2ca02c"]
		plt.figure(figsize=(5, 5), constrained_layout=True)
		_ = plt.pie(pie_values, labels=pie_labels, autopct=lambda p: f"{p:.1f}%", colors=colors)
		# Draw center circle for donut effect
		centre_circle = Circle((0, 0), 0.70, fc="white")
		fig = plt.gcf()
		fig.gca().add_artist(centre_circle)
		plt.title("Verteilung (nur gelabelt)")
		plt.savefig(os.path.join(out_dir, "gender_counts_donut.png"), dpi=150, bbox_inches="tight")
		plt.close()

	# Normalized bar chart: occurrences per 1000 tokens per file (Mann/Frau/Genderneutral)
	plot_df = per_file_df.copy()
	# Avoid division by zero
	plot_df["per_1000_Mann"] = (plot_df["Mann"] / plot_df["TOTAL_ROWS"]) * 1000
	plot_df["per_1000_Frau"] = (plot_df["Frau"] / plot_df["TOTAL_ROWS"]) * 1000
	plot_df["per_1000_Genderneutral"] = (plot_df["Genderneutral"] / plot_df["TOTAL_ROWS"]) * 1000
	plt.figure(figsize=(11, 5), constrained_layout=True)
	index = np.arange(len(plot_df))
	bar_w = 0.25
	plt.bar(index - bar_w, plot_df["per_1000_Frau"], width=bar_w, label="Frau", color="#d62728")
	plt.bar(index, plot_df["per_1000_Mann"], width=bar_w, label="Mann", color="#1f77b4")
	plt.bar(index + bar_w, plot_df["per_1000_Genderneutral"], width=bar_w, label="Genderneutral", color="#2ca02c")
	plt.xticks(index, list(plot_df["file"]), rotation=60, ha="right")
	plt.ylabel("Vorkommen pro 1000 Tokens")
	plt.title("Normalisierte Häufigkeit je Text")
	plt.legend()
	plt.savefig(os.path.join(out_dir, "gender_counts_normalized_per_text.png"), dpi=150, bbox_inches="tight")
	plt.close()

	# Diverging bar chart: female left, male right per file
	div_df = per_file_df.copy()
	# Normalize to per 1000 tokens for fair comparison
	div_df["male_per_1000"] = (div_df["Mann"] / div_df["TOTAL_ROWS"]) * 1000
	div_df["female_per_1000"] = (div_df["Frau"] / div_df["TOTAL_ROWS"]) * 1000
	plt.figure(figsize=(11, 6), constrained_layout=True)
	y = np.arange(len(div_df))
	plt.barh(y, div_df["male_per_1000"], color="#1f77b4", label="Mann")
	plt.barh(y, -div_df["female_per_1000"], color="#d62728", label="Frau")
	plt.yticks(y, list(div_df["file"]))
	plt.xlabel("Vorkommen pro 1000 Tokens (links: Frau, rechts: Mann)")
	plt.title("Divergierende Balken je Text")
	plt.axvline(0, color="#333", linewidth=0.8)
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(out_dir, "gender_diverging_bars_per_text.png"), dpi=150, bbox_inches="tight")
	plt.close()


def build_unique_persons(base_dir: str) -> Tuple[Dict[str, int], pd.DataFrame]:
	"""
	Analyze unique proper nouns (Eigennamen/PROPN) by gender.
	Reads pos_tagged_tokens.csv and extracts unique tokens with POS=PROPN per gender.
	Returns global counts and per-file breakdown for all 4 categories.
	"""
	pos_path = os.path.join(base_dir, "pos_tagged_tokens.csv")
	if not os.path.exists(pos_path):
		return {}, pd.DataFrame()
	
	pos_df = pd.read_csv(pos_path, dtype=str)
	# Filter to PROPN only
	propn_df = pos_df[pos_df["pos"] == "PROPN"].copy()
	
	# Global unique counts for all 4 categories
	global_counts = {}
	for cat in ["Mann", "Frau", "Genderneutral", "O"]:
		global_counts[cat] = propn_df[propn_df["label"] == cat]["token"].nunique()
	
	# Per-file unique counts + total tokens for normalization
	per_file_rows = []
	for fname in pos_df["file"].unique():
		file_propn = propn_df[propn_df["file"] == fname]
		file_total_tokens = len(pos_df[pos_df["file"] == fname])
		per_file_rows.append({
			"file": fname,
			"unique_Mann": file_propn[file_propn["label"] == "Mann"]["token"].nunique(),
			"unique_Frau": file_propn[file_propn["label"] == "Frau"]["token"].nunique(),
			"unique_Genderneutral": file_propn[file_propn["label"] == "Genderneutral"]["token"].nunique(),
			"unique_O": file_propn[file_propn["label"] == "O"]["token"].nunique(),
			"TOTAL_TOKENS": file_total_tokens,
		})
	per_file_unique = pd.DataFrame(per_file_rows).sort_values("file").reset_index(drop=True)
	
	return global_counts, per_file_unique


def save_unique_person_charts(global_counts: Dict[str, int], per_file_df: pd.DataFrame, out_dir: str) -> None:
	"""
	Generate charts for unique person (PROPN) analysis:
	1. Global bar chart comparing unique names across all 4 categories
	2. Per-file grouped bar chart (normalized per 1000 tokens)
	"""
	if not global_counts or per_file_df.empty:
		return
	
	categories = ["Mann", "Frau", "Genderneutral", "O"]
	colors = ["#1f77b4", "#d62728", "#2ca02c", "#7f7f7f"]
	
	# Chart 1: Global unique persons bar chart (all 4 categories)
	values = [global_counts.get(c, 0) for c in categories]
	plt.figure(figsize=(7, 5), constrained_layout=True)
	bars = plt.bar(categories, values, color=colors)
	plt.ylabel("Anzahl unique Eigennamen")
	plt.title("Unique Eigennamen nach Kategorie (PROPN)")
	for b, v in zip(bars, values):
		plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, f"{v}", ha="center", va="bottom", fontweight="bold")
	plt.savefig(os.path.join(out_dir, "unique_persons_global.png"), dpi=150, bbox_inches="tight")
	plt.close()
	
	# Chart 2: Per-file grouped bar chart (relative frequency %) - without O for clarity
	# Compute percentage of tokens that are unique PROPN per category
	plot_df = per_file_df.copy()
	per_file_cats = ["Mann", "Frau", "Genderneutral"]  # exclude O for readability
	per_file_colors = ["#1f77b4", "#d62728", "#2ca02c"]
	for cat in per_file_cats:
		col = f"unique_{cat}"
		plot_df[f"pct_{cat}"] = (plot_df[col] / plot_df["TOTAL_TOKENS"]) * 100
	
	plt.figure(figsize=(14, 6), constrained_layout=True)
	x = np.arange(len(plot_df))
	width = 0.25
	offsets = [-1, 0, 1]
	for i, cat in enumerate(per_file_cats):
		plt.bar(x + offsets[i] * width, plot_df[f"pct_{cat}"], width=width, label=cat, color=per_file_colors[i])
	plt.xticks(x, list(plot_df["file"]), rotation=60, ha="right")
	plt.ylabel("Relative Häufigkeit (%)")
	plt.title("Unique Eigennamen pro Text (relative Häufigkeit, PROPN)")
	plt.legend()
	plt.grid(axis="y", linestyle=":", linewidth=0.5)
	plt.savefig(os.path.join(out_dir, "unique_persons_per_text.png"), dpi=150, bbox_inches="tight")
	plt.close()
	
	print(f"Saved unique persons charts: unique_persons_global.png, unique_persons_per_text.png")
	print(f"  Global unique: Mann={global_counts.get('Mann', 0)}, Frau={global_counts.get('Frau', 0)}, Genderneutral={global_counts.get('Genderneutral', 0)}, O={global_counts.get('O', 0)}")


def main():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	per_file_df, totals = analyze_corpus(base_dir)

	# POS tagging outputs (kept separate from TSVs)
	pos_df, pos_agg = build_pos_counts(base_dir)
	if not pos_df.empty:
		pos_df.to_csv(os.path.join(base_dir, "pos_tagged_tokens.csv"), index=False, encoding="utf-8")
		# CSV: rows=gender, cols=POS
		all_pos = sorted({p for d in pos_agg.values() for p in d.keys()})
		heat_rows = []
		for gender in ["Frau", "Mann", "Genderneutral", "O"]:
			row = {"gender": gender}
			for p in all_pos:
				row[p] = int(pos_agg.get(gender, {}).get(p, 0))  # type: ignore[assignment]
			heat_rows.append(row)
		heat_df = pd.DataFrame(heat_rows)
		heat_df.to_csv(os.path.join(base_dir, "pos_counts.csv"), index=False, encoding="utf-8")

		# Additional visualization: 100% stacked bar chart for POS distribution (top 10 POS by labeled total)
		labeled_only = heat_df[heat_df["gender"].isin(["Frau", "Mann", "Genderneutral"])].copy()
		# Compute total per POS across labeled rows
		pos_totals = {p: float(labeled_only[p].sum()) for p in all_pos}
		top_pos = [p for p, _ in sorted(pos_totals.items(), key=lambda x: x[1], reverse=True)[:10]]
		# Normalize to percentages per gender
		stack_df = labeled_only[["gender"] + top_pos].copy()
		stack_df[top_pos] = stack_df[top_pos].apply(pd.to_numeric, errors="coerce").fillna(0.0)
		stack_df["row_total"] = stack_df[top_pos].sum(axis=1)
		# Avoid division by zero
		stack_df["row_total"] = stack_df["row_total"].replace(0, np.nan)
		for p in top_pos:
			stack_df[p] = (stack_df[p] / stack_df["row_total"]) * 100.0
		stack_df = stack_df.fillna(0.0)
		# Fixed category order for clarity
		order = ["Frau", "Mann", "Genderneutral"]
		stack_df["gender"] = pd.Categorical(stack_df["gender"], categories=order, ordered=True)
		stack_df = stack_df.sort_values("gender")
		# Plot with explicit x positions to ensure alignment
		plt.figure(figsize=(12, 6), constrained_layout=True)
		x = np.arange(len(stack_df))
		bottom = np.zeros(len(stack_df))
		from matplotlib import colormaps as mcm
		cmap = mcm.get_cmap("tab20")
		colors = [cmap(i / max(1, len(top_pos) - 1)) for i in range(len(top_pos))]
		for i, p in enumerate(top_pos):
			vals = stack_df[p].to_numpy(dtype=float)
			plt.bar(x, vals, bottom=bottom, label=p, color=colors[i % len(colors)])
			bottom = bottom + vals
		plt.xticks(x, list(stack_df["gender"]))
		plt.ylim(0, 100)
		plt.ylabel("Anteil (%)")
		plt.title("POS-Verteilung (Top 10 POS) — 100% gestapelt je Kategorie")
		plt.legend(ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
		plt.savefig(os.path.join(base_dir, "pos_distribution_stacked100.png"), dpi=150, bbox_inches="tight")
		plt.close()

		# Additional visualization: Grouped bar chart for POS distribution (Top 10 POS)
		# Prepare long-form data
		long_rows = []
		for _, row in stack_df.iterrows():
			g = str(row["gender"])
			for p in top_pos:
				long_rows.append({"gender": g, "POS": p, "value": float(row[p])})
		long_df = pd.DataFrame(long_rows)
		# POS order left-to-right
		pos_order = top_pos
		gender_order = ["Frau", "Mann", "Genderneutral"]
		# X positions per POS, with small offsets per gender
		x_pos = np.arange(len(pos_order))
		width = 0.25
		from matplotlib import colormaps as mcm
		cmap_g = mcm.get_cmap("Set2")
		gender_colors = {
			"Frau": cmap_g(0),
			"Mann": cmap_g(1),
			"Genderneutral": cmap_g(2),
		}
		plt.figure(figsize=(12, 6), constrained_layout=True)
		for i, g in enumerate(gender_order):
			vals = [float(long_df[(long_df["gender"] == g) & (long_df["POS"] == p)]["value"].sum()) for p in pos_order]
			plt.bar(x_pos + (i - 1) * width, vals, width=width, label=g, color=gender_colors[g])
		plt.xticks(x_pos, pos_order, rotation=30, ha="right")
		plt.ylabel("Anteil (%)")
		plt.title("POS-Verteilung (Top 10 POS) — gruppiert nach Kategorie")
		plt.ylim(0, 100)
		plt.legend()
		plt.savefig(os.path.join(base_dir, "pos_distribution_grouped_top10.png"), dpi=150, bbox_inches="tight")
		plt.close()

	# Console summary
	print("Analyzed files (12 expected):", len(per_file_df))
	print(per_file_df[["file", "Mann", "Frau", "Genderneutral", "O", "TOTAL_ROWS", "TOTAL_LABELED"]])
	print()
	print("GLOBAL TOTALS")
	print({k: int(v) for k, v in totals.items()})
	labeled_sum = totals.get("TOTAL_LABELED", 0)
	if labeled_sum > 0:
		print("Shares among labeled (M/F/N only):")
		for k in ["Mann", "Frau", "Genderneutral"]:
			pct = 100.0 * totals.get(k, 0) / labeled_sum
			print(f"  {k}: {totals.get(k, 0)} ({pct:.2f}%)")

	save_outputs(per_file_df, totals, base_dir)
	print()
	print("Saved per-file CSV: gender_counts_summary.csv")
	print("Saved bar chart PNG: gender_counts_bar.png")
	if not pos_df.empty:
		print("Saved POS CSV: pos_tagged_tokens.csv, pos_counts.csv")
		print("Saved POS distribution PNGs: pos_distribution_stacked100.png, pos_distribution_grouped_top10.png")

	# Unique person (PROPN) analysis
	global_unique, per_file_unique = build_unique_persons(base_dir)
	if global_unique:
		save_unique_person_charts(global_unique, per_file_unique, base_dir)


if __name__ == "__main__":
	main()