from flask import Flask, request, render_template
import os
import spacy
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import networkx as nx
from spacy import displacy
import matplotlib.pyplot as plt
import uuid
from flask import send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER= 'uploads'
STATIC_FOLDER= 'static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load your trained spaCy model (update the path accordingly)
model_directory = 'G:/My Drive/5. Research/2. Phd Research/3.0 Research/NER/Web_tool_NER/Agri_Bio_NER/model-best_spacy_PMBert_Fltxt_bnry'
nlp = spacy.load(model_directory)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(STATIC_FOLDER, filename, as_attachment=True)

@app.route("/extract_entities", methods=["POST"])
def extract_entities():
    text = request.form.get("text", "").strip()
    if not text:
        return "No text provided!", 400
    return process_text(text)

@app.route("/predict_file", methods=["POST"])
def predict_file():
    text_file = request.files.get("text_file")
    
    if not text_file or text_file.filename == "":
        return "Error: No file uploaded", 400

    # Secure the filename and save the file
    filename = secure_filename(text_file.filename)
    text_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    text_file.save(text_path)
    print(f"File saved to: {text_path}")

    # Read the content of the saved file
    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except UnicodeDecodeError:
        return "File decoding failed. Ensure it is a UTF-8 encoded text file.", 400

    if not text:
        return "Empty file content", 400

    # Call your processing function
    return process_text(text)


# Common processing function
def process_text(text):
    doc = nlp(text)

    colors = {
        "DISEASE": "linear-gradient(90deg, #ff9999, #ff4d4d)",
        "NON-CODING_RNA": "linear-gradient(90deg, #cc66ff, #9933cc)",
    }
    options = {"ents": list(colors.keys()), "colors": colors, "bg": "rgba(0,0,0,0)"}

    highlighted_text = displacy.render(doc, style="ent", options=options, manual=False)

    # Build label->entities dictionary (unique entities)
    label_dict = {}
    for ent in doc.ents:
        label_dict.setdefault(ent.label_, set()).add(ent.text.strip())

     # Define label-specific colors
    label_colors = {
    'NON-CODING_RNA': 'darkgreen',
    'DISEASE': 'crimson',
    # Add more label-color pairs as needed
    }

    data = []
    for label, entities in label_dict.items():
        entity_links = []
        for entity in sorted(entities):
            url = f"https://www.google.com/search?q={entity.replace(' ', '+')}"
            color = label_colors.get(label, 'black')  # Default to black if label not found
            styled_entity = f'<a href="{url}" target="_blank"><span style="color: {color};">{entity}</span></a>'
            entity_links.append(styled_entity)
        data.append([label, ', '.join(entity_links)])

    df = pd.DataFrame(data, columns=['Label', 'Entities'])
    df = df.sort_values(by='Label').reset_index(drop=True)

    # Convert to HTML with alignment
    entity_link_table = df.to_html(
        escape=False,
        index=False,
        classes='custom-table'
    )

    # Add CSS to align Label left and Entities right
    custom_css = """
    <style>
        .custom-table {
            border-collapse: collapse;
            width: 100%;
        }

        .custom-table th, .custom-table td {
            border: 1px solid #ddd;
            padding: 8px;
        }

        .custom-table th {
            background-color: #f2f2f2;
        }

        .custom-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .custom-table th:first-child, .custom-table td:first-child {
            text-align: left;
        }

        .custom-table th:last-child, .custom-table td:last-child {
            text-align: right;
        }
    </style>
    """
    
    # Final HTML with style
    entity_link_table = custom_css + entity_link_table
    
    entity_freq = Counter(ent.text.strip().replace("\n", " ") for ent in doc.ents)
    wordcloud_filename = f"wordcloud_{uuid.uuid4()}.png"
    wordcloud_path = os.path.join(STATIC_FOLDER, wordcloud_filename)
    generate_wordcloud(entity_freq, wordcloud_path)
    wordcloud_url = '/' + wordcloud_path.replace("\\", "/")

    label_dict = {}
    for ent in doc.ents:
        label_dict.setdefault(ent.label_, set()).add(ent.text)
    network_filename = f"network_{uuid.uuid4()}.png"
    network_path = os.path.join(STATIC_FOLDER, network_filename)
    generate_network_image(label_dict, network_path)
    network_url = '/' + network_path.replace("\\", "/")

    return render_template(
        "result.html",
        highlighted_text=highlighted_text,
        entity_link_table=entity_link_table,
        wordcloud_url=wordcloud_url,
        network_url=network_url
    )

# Wordcloud generator
def generate_wordcloud(freq_dict, output_path):
    wc = WordCloud(width=1600, height=800, background_color="white", colormap='tab10')
    wc.generate_from_frequencies(freq_dict)
    plt.figure(figsize=(16, 8), dpi=300)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate Network One Below Another
def generate_network_image(label_dict, output_path):
    labels = list(label_dict.keys())
    if len(labels) != 2:
        raise ValueError("This function only works when exactly two labels are present.")

    fig, axes = plt.subplots(2, 1, figsize=(14, 28), dpi=300)  # 2 rows, 1 column

    for i, label in enumerate(labels):
        G = nx.Graph()
        G.add_node(label, color='lightblue', size=800)
        for ent in label_dict[label]:
            G.add_node(ent, color='lightpink', size=400)
            G.add_edge(label, ent)

        pos = nx.spring_layout(G, k=0.5, iterations=100)
        colors = [G.nodes[n]['color'] for n in G.nodes()]
        sizes = [G.nodes[n]['size'] for n in G.nodes()]

        nx.draw(
            G, pos, with_labels=True, node_color=colors, node_size=sizes,
            font_size=14, font_weight='bold', edge_color='gray', ax=axes[i]
        )
        axes[i].set_title(f"{label} Network", fontsize=16, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    app.run(debug=True)
