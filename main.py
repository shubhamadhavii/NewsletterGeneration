from flask import Flask, render_template, url_for,request
app = Flask(__name__)
from transformers import GPT2LMHeadModel, GPT2Tokenizer



@app.route("/home")
@app.route('/')
def index():
    # 18. Now add route index
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 7. Explain the render template and import and also how it finds the index
    # 8. Also explain how we need to create a template now
    title = request.form['title']
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id)
    input_sentence =title
    input_ids = tokenizer.encode(input_sentence, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    output_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

    return render_template('index.html', text=output_sentence)


if __name__ == '__main__':
    app.run(debug=True)
