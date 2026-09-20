import math
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Complete AI Classroom", page_icon="🧠", layout="wide")
np.random.seed(7)

st.markdown("""
<style>
.main .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
.hero {padding:1.25rem 1.5rem;border-radius:18px;background:linear-gradient(120deg,#172554,#312e81);color:white;margin-bottom:1rem}
.card {padding:1rem 1.1rem;border:1px solid #dbe4f0;border-radius:14px;background:#f8fafc;margin:.5rem 0}
.term {font-weight:700;color:#312e81}.small {font-size:.9rem;color:#475569}
</style>
""", unsafe_allow_html=True)


def card(title, text, example=None):
    extra = f"<div class='small'><b>Example:</b> {example}</div>" if example else ""
    st.markdown(f"<div class='card'><div class='term'>{title}</div>{text}{extra}</div>", unsafe_allow_html=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def softmax(x):
    z = np.asarray(x) - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def tokenize(text):
    return re.findall(r"[a-z0-9']+|[^\w\s]", text.lower())


def bow(text, vocab):
    c = Counter(tokenize(text))
    return np.array([c[w] for w in vocab], dtype=float)


st.markdown("<div class='hero'><h1>🧠 Complete Interactive AI Classroom</h1><p>From rules and regression to attention, RAG, LoRA and AI agents—small simulations that reveal what happens underneath.</p></div>", unsafe_allow_html=True)

MODULES = [
    "Home & Teaching Guide",
    "1 · Artificial Intelligence",
    "2 · Machine Learning",
    "3 · Deep Learning",
    "4 · LLMs & Generative AI",
    "Glossary & Revision Quiz",
]
page = st.sidebar.radio("Choose a module", MODULES)
st.sidebar.markdown("---")
st.sidebar.caption("Runs locally • No API key • No GPU • No internet required")


if page == "Home & Teaching Guide":
    st.header("How to use this app")
    st.write("Use the sidebar to teach one module at a time. Every demo follows **idea → experiment → observation → takeaway**. Change a control, ask students to predict the result, and then run it.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modules", 4); c2.metric("Concepts", "50+"); c3.metric("Interactive demos", "20+"); c4.metric("External APIs", 0)
    st.subheader("Recommended 90-minute route")
    route = pd.DataFrame({"Minutes":[15,25,20,30],"Module":["AI","Machine Learning","Deep Learning","LLMs"],"Best live demos":["Rules & CV","Regression, clustering, Q-learning","Neuron, CNN, RNN","Tokens, attention, RAG, agents"]})
    st.dataframe(route, hide_index=True, use_container_width=True)
    st.info("This is a conceptual simulator. Real production systems use larger datasets, many layers, GPUs, evaluation suites and safety controls.")
    st.subheader("The big picture")
    st.code("AI ⊃ Machine Learning ⊃ Deep Learning ⊃ many modern Generative-AI systems", language=None)
    card("Artificial Intelligence", "The broad goal: machines performing tasks associated with intelligent behavior.")
    card("Machine Learning", "Systems learn patterns from data rather than receiving every rule explicitly.")
    card("Deep Learning", "Machine learning with layered neural networks that learn useful representations.")
    card("Generative AI", "Models learn a data distribution and create new text, images, audio, code or video.")


elif page == "1 · Artificial Intelligence":
    st.header("1. Artificial Intelligence")
    tab1, tab2, tab3, tab4 = st.tabs(["Scope of AI", "Rules & Expert Systems", "Computer Vision", "NLP & Robotics"])
    with tab1:
        card("Artificial Narrow Intelligence (ANI)", "Excels at a bounded task but does not possess general human capability.", "spam filter, chess engine, recommendation system")
        card("Artificial General Intelligence (AGI)", "A hypothetical system able to learn and perform broadly across intellectual tasks at roughly human breadth. No universally accepted test or confirmed AGI exists.")
        card("Artificial Superintelligence (ASI)", "A hypothetical intelligence exceeding humans across most cognitive domains. It is a future concept, not a current product.")
        st.warning("ANI, AGI and ASI describe **scope of capability**, not three algorithms. A fluent chatbot should not automatically be called AGI.")
    with tab2:
        st.subheader("Symbolic / rule-based AI")
        temp = st.slider("Temperature (°C)", 34.0, 42.0, 38.2, .1)
        cough = st.checkbox("Cough", True); rash = st.checkbox("Rash")
        rules = []
        if temp >= 38: rules.append("IF temperature ≥ 38 THEN fever")
        if temp >= 38 and cough: rules.append("IF fever AND cough THEN possible respiratory infection")
        if rash: rules.append("IF rash THEN dermatology review")
        if not rules: rules.append("No rule fired")
        st.code("\n".join(rules), language=None)
        card("Expert system", "A knowledge base of domain facts/rules plus an inference engine that applies them. It can explain which rule fired, but it is brittle outside encoded knowledge.")
        st.caption("Educational only—this is not medical advice or a diagnostic system.")
    with tab3:
        st.subheader("Computer vision: an image is a grid of numbers")
        img = np.zeros((8,8)); img[2:6,3:5]=1; img[3:5,2:6]=1
        noise = st.slider("Image noise", 0.0, .8, .15, .05)
        noisy = np.clip(img + np.random.default_rng(4).normal(0, noise, img.shape), 0, 1)
        fig, ax = plt.subplots(1,2,figsize=(6,2.7)); ax[0].imshow(img,cmap="gray",vmin=0,vmax=1); ax[0].set_title("Clean pixels"); ax[1].imshow(noisy,cmap="gray",vmin=0,vmax=1); ax[1].set_title("Camera input")
        for a in ax:a.axis("off")
        st.pyplot(fig)
        card("Computer Vision (CV)", "Algorithms extract meaning from pixels: classification, detection, segmentation, OCR and generation. A model learns visual patterns; it does not see as a human does.")
    with tab4:
        text = st.text_input("Sentence for a tiny NLP pipeline", "Robots can understand simple commands")
        toks = tokenize(text)
        st.write("Tokens:", toks)
        st.write("Simple intent:", "MOVE" if any(w in toks for w in ["move","go","walk"]) else "INFORMATION")
        card("Natural Language Processing (NLP)", "Methods for processing, understanding and generating human language: search, translation, sentiment, extraction and dialogue.")
        card("Robotics", "Combines perception, planning and physical action. Sensors observe; a controller decides; actuators move; feedback closes the loop.", "warehouse robot avoids an obstacle and replans")


elif page == "2 · Machine Learning":
    st.header("2. Machine Learning")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Supervised", "Unsupervised", "Reinforcement", "Training mechanics", "Generalization"])
    with tab1:
        task = st.radio("Task", ["Regression", "Classification"], horizontal=True)
        rng=np.random.default_rng(2); x=np.linspace(1,10,30); y=5+7*x+rng.normal(0,5,30)
        if task == "Regression":
            m = st.slider("Slope parameter (weight)", 0.0, 12.0, 6.0, .2); b=st.slider("Bias", -5.0, 15.0, 4.0,.5)
            pred=m*x+b; loss=np.mean((pred-y)**2)
            fig,ax=plt.subplots(figsize=(7,3)); ax.scatter(x,y,label="training examples"); ax.plot(x,pred,color="crimson",label="prediction"); ax.set(xlabel="Study hours (feature)",ylabel="Score (target)"); ax.legend(); st.pyplot(fig)
            st.metric("Mean squared error loss", f"{loss:.2f}")
            st.write("Regression predicts a continuous number. The **parameters** are slope and bias; changing them changes every prediction.")
        else:
            threshold=st.slider("Decision threshold", 30,85,55); scores=np.array([32,45,51,58,63,72,88]); labels=np.array([0,0,0,1,1,1,1]); preds=(scores>=threshold).astype(int)
            st.dataframe(pd.DataFrame({"score (feature)":scores,"passed (target)":labels,"prediction":preds}),hide_index=True)
            st.metric("Accuracy",f"{(preds==labels).mean():.0%}")
            st.write("Classification predicts a category. A probability threshold converts a score into a class.")
        st.info("**Feature:** input used for prediction. **Target/label:** desired output. **Parameter:** learned value. **Hyperparameter:** choice set before training, such as learning rate or tree depth.")
    with tab2:
        demo=st.selectbox("Demo",["Clustering","Dimensionality reduction (PCA)","Anomaly detection"])
        rng=np.random.default_rng(5); a=rng.normal([2,2],[.5,.7],(30,2)); b=rng.normal([6,5],[.7,.5],(30,2)); X=np.vstack([a,b])
        if demo=="Clustering":
            centers=np.array([[2.,5.],[6.,2.]])
            for _ in range(st.slider("K-means iterations",1,10,4)):
                lab=np.argmin(((X[:,None,:]-centers[None,:,:])**2).sum(2),axis=1)
                centers=np.array([X[lab==k].mean(0) for k in range(2)])
            fig,ax=plt.subplots(figsize=(6,3)); ax.scatter(X[:,0],X[:,1],c=lab,cmap="coolwarm"); ax.scatter(centers[:,0],centers[:,1],s=180,marker="X",c="black"); st.pyplot(fig)
            st.write("Clustering discovers groups without labels. K-means alternates assignment and center-update steps.")
        elif demo.startswith("Dimensionality"):
            X3=np.c_[X, X[:,0]*.7+X[:,1]*.2+np.random.default_rng(1).normal(0,.2,len(X))]; Z=X3-X3.mean(0); _,s,v=np.linalg.svd(Z,full_matrices=False); proj=Z@v[:2].T
            st.write("3 original features → 2 principal components")
            st.dataframe(pd.DataFrame(proj[:8],columns=["PC1","PC2"]),hide_index=True)
            st.metric("Variance retained",f"{((s[:2]**2).sum()/(s**2).sum()):.1%}")
            st.write("PCA rotates data toward directions with most variance, then keeps fewer directions. Components are combinations—not original columns.")
        else:
            X2=np.vstack([X,[[10,0]]]); center=X2.mean(0); dist=np.linalg.norm(X2-center,axis=1); cut=np.quantile(dist,st.slider("Anomaly percentile",.80,.98,.93,.01)); flag=dist>cut
            fig,ax=plt.subplots(figsize=(6,3)); ax.scatter(X2[:,0],X2[:,1],c=np.where(flag,"crimson","steelblue")); st.pyplot(fig)
            st.write(f"Flagged {flag.sum()} unusually distant point(s). Anomaly detection learns what is normal and identifies rare deviations; rare does not always mean wrong.")
    with tab3:
        st.subheader("Q-learning in a five-room corridor")
        episodes=st.slider("Training episodes",5,500,120,5); alpha=st.slider("Learning rate α",.05,1.0,.4,.05); gamma=st.slider("Future reward γ",0.0,.99,.9,.05)
        Q=np.zeros((5,2)); rng=np.random.default_rng(8)
        for ep in range(episodes):
            s=0
            for _ in range(20):
                act=rng.integers(2) if rng.random()<max(.05,1-ep/max(episodes,1)) else np.argmax(Q[s])
                ns=max(0,s-1) if act==0 else min(4,s+1); reward=10 if ns==4 else -0.1
                Q[s,act]+=alpha*(reward+gamma*np.max(Q[ns])-Q[s,act]); s=ns
                if s==4:break
        st.dataframe(pd.DataFrame(Q,columns=["Q(left)","Q(right)"],index=[f"State {i}" for i in range(5)]).style.format("{:.2f}"))
        st.success("Learned policy: "+" → ".join("RIGHT" if np.argmax(Q[s]) else "LEFT" for s in range(4))+" → GOAL")
        card("Markov Decision Process (MDP)", "Formalizes states, actions, transition probabilities, rewards and discounting. The Markov assumption says the current state contains the information needed for the next transition.")
        card("Q-learning", "Learns the long-term value Q(state, action) using reward plus the best estimated future value. Exploration gathers experience; exploitation uses the best-known action.")
    with tab4:
        st.subheader("Gradient descent learns one weight")
        lr=st.slider("Learning rate (hyperparameter)",.001,.2,.03,.001); steps=st.slider("Optimization steps",1,100,30)
        x=np.array([1.,2.,3.,4.]); y=3*x; w=0.; hist=[]
        for _ in range(steps):
            pred=w*x; loss=np.mean((pred-y)**2); grad=2*np.mean((pred-y)*x); w-=lr*grad; hist.append(loss)
        c1,c2=st.columns(2); c1.metric("Learned weight",f"{w:.3f}","target 3.000"); c2.metric("Final MSE loss",f"{hist[-1]:.5f}")
        fig,ax=plt.subplots(figsize=(6,2.5)); ax.plot(hist); ax.set(xlabel="step",ylabel="loss"); st.pyplot(fig)
        st.write("A **loss function** measures error. An **optimizer** changes parameters in the direction that reduces loss. Too small a learning rate is slow; too large can overshoot or diverge.")
        st.write("Common losses: MSE for regression; cross-entropy for classification; task-specific ranking or contrastive losses for other objectives.")
    with tab5:
        degree=st.slider("Polynomial complexity",1,12,2); rng=np.random.default_rng(3); xt=np.linspace(-1,1,18); yt=np.sin(3*xt)+rng.normal(0,.18,len(xt)); coef=np.polyfit(xt,yt,degree); xx=np.linspace(-1.2,1.2,200); yy=np.polyval(coef,xx)
        fig,ax=plt.subplots(figsize=(7,3)); ax.scatter(xt,yt,label="training data"); ax.plot(xx,yy,color="crimson",label=f"degree {degree}"); ax.plot(xx,np.sin(3*xx),"--",label="true pattern"); ax.set_ylim(-2,2); ax.legend(); st.pyplot(fig)
        card("Underfitting", "Model is too simple: high training and test error (high bias).")
        card("Overfitting", "Model memorizes noise: low training error but poor new-data performance (high variance).")
        card("Bias–variance trade-off", "More flexibility can reduce systematic error but increase sensitivity to the sample. Validation data, regularization and more representative data help select a useful balance.")


elif page == "3 · Deep Learning":
    st.header("3. Deep Learning")
    tab1,tab2,tab3,tab4=st.tabs(["Neuron & ANN","Activations & Backprop","CNN","RNN & LSTM"])
    with tab1:
        x1=st.slider("Input x₁",-2.0,2.0,.8,.1); x2=st.slider("Input x₂",-2.0,2.0,-.3,.1); w1=st.slider("Weight w₁",-3.0,3.0,1.2,.1); w2=st.slider("Weight w₂",-3.0,3.0,-.7,.1); bias=st.slider("Bias b",-2.0,2.0,.1,.1)
        z=x1*w1+x2*w2+bias; out=sigmoid(z)
        st.latex(r"z=x_1w_1+x_2w_2+b")
        st.latex(fr"z=({x1:.1f})({w1:.1f})+({x2:.1f})({w2:.1f})+{bias:.1f}={z:.2f}\quad\Rightarrow\quad sigmoid(z)={out:.3f}")
        st.progress(float(out)); st.caption("A node combines inputs with weights and bias, then applies an activation.")
        card("ANN", "A network of connected artificial neurons. Input, hidden and output layers transform data.")
        card("DNN", "An ANN with multiple hidden layers. Depth enables hierarchical representations, but makes optimization and interpretation harder.")
    with tab2:
        act=st.selectbox("Activation",["ReLU","Sigmoid","Softmax"]); vals=np.array([-2.,-.5,.5,2.])
        if act=="ReLU": res=np.maximum(0,vals); explanation="Keeps positive values and sets negatives to zero; common in hidden layers."
        elif act=="Sigmoid":res=sigmoid(vals); explanation="Maps values to 0–1; useful for binary output probabilities but can saturate."
        else:res=softmax(vals); explanation="Turns a vector of logits into class probabilities summing to one."
        st.dataframe(pd.DataFrame({"input/logit":vals,"output":res}),hide_index=True); st.write(explanation)
        target=st.slider("Target y",0.0,1.0,1.0,.1); weight=st.slider("Current weight",-2.0,2.0,.2,.1); inp=.8; pred=sigmoid(weight*inp); loss=.5*(pred-target)**2; grad=(pred-target)*pred*(1-pred)*inp
        st.write(f"Forward pass: prediction **{pred:.3f}**, loss **{loss:.4f}**")
        st.write(f"Backward pass: chain-rule gradient dLoss/dWeight = **{grad:.4f}**")
        st.code(f"new_weight = {weight:.3f} - learning_rate × ({grad:.4f})",language=None)
        card("Backpropagation", "Efficiently applies the chain rule from output back through layers, assigning each parameter its contribution to error. The optimizer then updates parameters.")
    with tab3:
        st.subheader("CNN-style edge detection")
        img=np.zeros((8,8)); img[:,4:]=1
        kernels={"Vertical edge":[[-1,0,1],[-1,0,1],[-1,0,1]],"Horizontal edge":[[-1,-1,-1],[0,0,0],[1,1,1]],"Blur":[[1/9]*3]*3}
        name=st.selectbox("Kernel/filter",list(kernels)); k=np.array(kernels[name]); out=np.zeros((6,6))
        for i in range(6):
            for j in range(6):out[i,j]=np.sum(img[i:i+3,j:j+3]*k)
        fig,ax=plt.subplots(1,3,figsize=(8,2.5)); ax[0].imshow(img,cmap="gray"); ax[0].set_title("Input"); ax[1].imshow(k,cmap="coolwarm"); ax[1].set_title("3×3 kernel"); ax[2].imshow(out,cmap="coolwarm"); ax[2].set_title("Feature map")
        for a in ax:a.axis("off")
        st.pyplot(fig)
        card("Convolutional Neural Network (CNN)", "Slides learned filters across spatial input. Shared weights detect local features such as edges; deeper layers combine them into shapes and objects. Pooling or strides can reduce size.")
    with tab4:
        sequence=st.text_input("Short sequence", "AI learns from data")
        tokens=tokenize(sequence)[:10]; h=0.; rows=[]
        for t in tokens:
            x=(sum(map(ord,t))%20)/10-1; h=np.tanh(.8*h+x); rows.append((t,x,h))
        st.dataframe(pd.DataFrame(rows,columns=["token","encoded input","RNN hidden state"]),hide_index=True)
        card("Recurrent Neural Network (RNN)", "Processes a sequence step by step and carries a hidden state. Long sequences can cause vanishing/exploding gradients and forgotten early information.")
        keep=st.slider("LSTM forget-gate value",0.0,1.0,.8,.05); old=st.slider("Old cell memory",-1.0,1.0,.7,.1); new=st.slider("New candidate memory",-1.0,1.0,.4,.1); write=st.slider("Input-gate value",0.0,1.0,.6,.05); cell=keep*old+write*new
        st.latex(fr"new\ cell = forget\ gate\times old + input\ gate\times candidate = {keep:.2f}\times{old:.2f}+{write:.2f}\times{new:.2f}={cell:.2f}")
        card("LSTM", "An RNN variant whose forget, input and output gates control a persistent cell state. This gives gradients and information a better path across time.")


elif page == "4 · LLMs & Generative AI":
    st.header("4. Large Language Models & Generative AI")
    tab1,tab2,tab3,tab4,tab5,tab6=st.tabs(["Tokens & Embeddings","Attention & Transformer","Training & Alignment","RAG","Agents & Multimodal","Prompts & Hallucination"])
    with tab1:
        text=st.text_area("Text", "Transformers learn relationships between tokens.")
        toks=tokenize(text); st.write("Toy tokens:",toks); st.metric("Token count",len(toks))
        st.caption("Real tokenizers often use subword units, so one token is not necessarily one word.")
        vocab=sorted(set(toks)); vectors=[]
        for t in toks:
            rng=np.random.default_rng(sum(map(ord,t))); vectors.append(rng.normal(0,1,4))
        if toks: st.dataframe(pd.DataFrame(vectors,index=toks,columns=["d1","d2","d3","d4"]).style.format("{:.2f}"))
        card("Embedding", "A learned dense vector. During training, vectors are shaped so useful linguistic or semantic relationships become accessible to the model.")
        window=st.slider("Context window (toy limit in tokens)",3,30,10); st.write("Visible to model:",toks[-window:]); st.write("Outside window:",toks[:-window])
        card("Context window", "Maximum token sequence processed at once. Content outside it must be truncated, summarized or retrieved; a larger window does not guarantee perfect recall.")
    with tab2:
        words=tokenize(st.text_input("Sentence", "the student solved the problem"))[:8]
        if words:
            E=np.array([np.random.default_rng(sum(map(ord,w))).normal(size=4) for w in words]); Q=E; K=E@np.array([[.8,.1,0,0],[.1,.9,0,0],[0,0,.7,.2],[0,0,.2,.8]])
            scores=Q@K.T/math.sqrt(4); A=np.array([softmax(r) for r in scores])
            focus=st.selectbox("Query token",words); idx=words.index(focus)
            st.bar_chart(pd.DataFrame({"attention":A[idx]},index=words))
            st.write("Attention weights sum to",round(A[idx].sum(),3))
        card("Self-attention", "Each token creates query, key and value vectors. Query–key similarity produces weights; a weighted sum of values brings relevant context into each token representation.")
        card("Transformer architecture", "Stacks multi-head self-attention and feed-forward blocks with residual connections and normalization. Positional information represents order. Decoder-only transformers predict the next token; other variants encode or transform sequences.")
        st.code("tokens → embeddings + positions → [attention → feed-forward] × N → logits → probabilities → next token",language=None)
        card("Generative AI", "Models that create new samples resembling patterns in training data. Generation usually samples or selects repeatedly from predicted output distributions.")
    with tab3:
        choice=st.radio("Concept",["Pre-training","Fine-tuning & PEFT","LoRA","RLHF","DPO"],horizontal=True)
        if choice=="Pre-training":
            st.write("The model learns broad statistical patterns from a very large corpus, commonly through next-token prediction. This is expensive and produces a base model—not automatically a helpful assistant.")
            phrase="AI models learn"; options={"patterns":.55,"quickly":.20,"bananas":.03,"from":.22}; st.bar_chart(pd.DataFrame({"next-token probability":options}))
        elif choice=="Fine-tuning & PEFT":
            st.write("**Full fine-tuning** updates most or all model weights. **Parameter-efficient fine-tuning (PEFT)** trains a small set of added or selected parameters, lowering compute and storage needs.")
            st.metric("Illustrative full update","7,000,000,000 parameters"); st.metric("Illustrative PEFT update","7,000,000 parameters","0.1% trainable")
        elif choice=="LoRA":
            rank=st.slider("Low rank r",1,16,2); d=64; full=d*d; lora=2*d*rank
            st.metric("Full square update parameters",full); st.metric("LoRA A+B parameters",lora,f"{lora/full:.1%} of full")
            st.latex(r"W' = W + BA")
            st.write("LoRA freezes the original matrix W and learns two small low-rank matrices A and B. Multiple adapters can specialize one base model.")
        elif choice=="RLHF":
            st.write("1. Humans rank model answers. 2. A reward model learns those preferences. 3. Reinforcement learning updates the policy to earn reward while constraining excessive drift from the reference model.")
            helpful=st.slider("Human helpfulness rating",1,5,4); safe=st.slider("Human safety rating",1,5,5); st.metric("Toy reward",f"{.6*helpful+.4*safe:.1f} / 5")
        else:
            chosen=st.slider("Log probability of preferred answer",-5.0,0.0,-.8,.1); rejected=st.slider("Log probability of rejected answer",-5.0,0.0,-2.0,.1); beta=.5; pref=sigmoid(beta*(chosen-rejected))
            st.metric("Toy preference probability",f"{pref:.1%}")
            st.write("DPO learns directly from preferred/rejected pairs relative to a reference policy, avoiding a separate learned reward model and online RL loop in the standard formulation.")
    with tab4:
        docs=["The AI laboratory is in Block C and opens at 9 AM.","The machine-learning exam is on Friday in Room 204.","Students can borrow GPUs through the computing centre.","The robotics club meets every Wednesday evening.","Library access requires a student identity card."]
        q=st.text_input("Ask the college knowledge base", "When does the robotics club meet?")
        vocab=sorted(set(tokenize(" ".join(docs+[q])))); qv=bow(q,vocab); sims=[cosine(qv,bow(d,vocab)) for d in docs]; order=np.argsort(sims)[::-1]
        top=st.slider("Retrieved chunks (top-k)",1,4,2)
        res=pd.DataFrame({"similarity":[sims[i] for i in order[:top]],"retrieved text":[docs[i] for i in order[:top]]}); st.dataframe(res,hide_index=True,use_container_width=True)
        st.markdown("**Context sent to the generator**")
        st.code("\n".join(docs[i] for i in order[:top]),language=None)
        st.success("Toy grounded answer: "+docs[order[0]])
        card("Retrieval-Augmented Generation (RAG)", "At query time: retrieve relevant external chunks, add them to the prompt, then generate. RAG can improve freshness and traceability but retrieval can fail and the generator can still hallucinate.")
    with tab5:
        col1,col2=st.columns(2)
        with col1:
            st.subheader("Multimodal AI")
            modality=st.multiselect("Available inputs",["Text","Image","Audio","Video","Sensor data"],["Text","Image"])
            st.write("Combined representation:"," + ".join(modality) if modality else "No input")
            st.write("Multimodal models connect information across formats—for example answering a question about an image or generating speech from text.")
        with col2:
            st.subheader("AI agent & tool use")
            request=st.text_input("User request","Calculate 17 × 24")
            if re.search(r"\d+\s*[+*×/-]\s*\d+",request): tool="Calculator"; action="Compute exact arithmetic"
            elif any(w in request.lower() for w in ["latest","weather","today"]):tool="Web/search";action="Retrieve current information"
            elif "email" in request.lower():tool="Email";action="Draft/request confirmation before sending"
            else:tool="No external tool";action="Answer from context/model knowledge"
            st.metric("Selected tool",tool); st.write("Planned action:",action)
            st.write("An agent combines a model with goals, memory/state and tools in a loop: **observe → plan → act → inspect → continue/stop**. Permissions and confirmation matter for consequential actions.")
    with tab6:
        prompt=st.text_area("Prompt", "Explain neural networks")
        role=st.selectbox("Role/context",["None","You are a patient college lecturer","You are a concise technical reviewer"]); format_=st.selectbox("Output format",["Unspecified","Three bullets","Analogy + example","Table"]); audience=st.selectbox("Audience",["Unspecified","First-year student","Software engineer","School student"])
        improved=f"{'' if role=='None' else role+'. '}Task: {prompt.strip()}. {'' if audience=='Unspecified' else 'Audience: '+audience+'. '}{'' if format_=='Unspecified' else 'Format: '+format_+'. '}State assumptions and do not invent facts."
        st.code(improved,language=None)
        card("Prompt engineering", "Designing instructions and context so the model has a clear task, audience, constraints, format and examples. Good prompting guides behavior; it does not add missing knowledge or guarantee truth.")
        conf=st.slider("Model confidence in an unsupported claim",0,100,88)
        st.error(f"Hallucination example: a fluent claim with {conf}% apparent confidence but no supporting evidence.")
        card("Hallucination", "Plausible but unsupported or false output. Reduce risk with authoritative context, retrieval, citations, constrained outputs, tool checks and human review. Confidence of wording is not factual confidence.")


else:
    st.header("Glossary & revision quiz")
    terms={"AI":"Broad field of machine intelligence","Feature":"Input variable","Target":"Desired output","Parameter":"Learned value","Hyperparameter":"Training choice set by practitioner","Loss":"Numeric training error","Gradient":"Direction and sensitivity of loss change","Embedding":"Learned dense vector","Token":"Model's unit of text","Attention":"Context-dependent weighted information mixing","RAG":"Retrieval plus generation","LoRA":"Low-rank parameter-efficient adaptation","Agent":"Model-driven loop that may use tools"}
    st.dataframe(pd.DataFrame(terms.items(),columns=["Term","Meaning"]),hide_index=True,use_container_width=True)
    st.subheader("Five-question check")
    questions=[("Which task predicts a continuous house price?",["Classification","Regression","Clustering"],"Regression"),("Which value is normally learned?",["Weight","Learning rate","Epoch count"],"Weight"),("What does RAG add before generation?",["Retrieved context","More model layers","Robot sensors"],"Retrieved context"),("What converts logits into class probabilities summing to 1?",["ReLU","Softmax","MSE"],"Softmax"),("What is a hallucination?",["A slow GPU","Unsupported model output","A large context window"],"Unsupported model output")]
    score=0
    for i,(q,opts,ans) in enumerate(questions):
        a=st.radio(q,["Choose…"]+opts,key=f"q{i}")
        if a==ans:score+=1
        elif a!="Choose…":st.caption(f"Answer: {ans}")
    st.metric("Score",f"{score} / {len(questions)}")

st.markdown("---")
st.caption("Complete AI Classroom • Educational simulator • Values and architectures are deliberately simplified for teaching")
