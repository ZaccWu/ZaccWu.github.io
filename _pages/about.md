---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

Hi, I'm **Ziyue (Zacc) Wu** (武子越), a 3rd-year Ph.D. student in the **School of Management, Zhejiang University**, advised by Professor [Xi Chen](https://person.zju.edu.cn/chenxi/). I received my **Bachelor's Degree with Dual Majors** in Statistics (STATS) and Information Systems (IS) at Chu Kochen Honors College, Zhejiang University, in 2021.  

I'm enthusiastic about the **interdisciplinary fields of <font color="purple">Statistics, Machine Learning, and Business Studies</font>**, and devoted to data-driven managerial practices. My research aims to address crucial challenges in both academia and practical applications, drawing cutting-edge ideas in the field of **Statistics** and **Computer Science** to design novel methodological solutions adapted to these challenges. I also seek to unveil new mechanisms and principles in business scenarios, contributing to the advancement of managerial theories.

Specifically, my research interests broadly include the following topics:

- **<font color="purple">Methodological Research</font>**: Casual inference in machine learning, Graph learning,
Time series forecasting, Interpretable machine learning, Bayesian statistics, and Bayesian deep learning.
- **<font color="purple">Applied Research</font>**: Social network, Social media, Customer behavior, Human-AI collaboration, Precision marketing and product promotion, and Financial technology.

**If you are seeking any form of academic cooperation, please feel free to email me at <font color="purple">ziyuewu@zju.edu.cn</font>**.

I also collaborate with companies for **academic research and practical implementations**, most of which are large online E-commerce or social media platforms in China such as Alibaba. My Ph.D. thesis plans to focus on social dynamic modeling, particularly the methodological development and mechanism discovery of individual behaviors in networks.



# 🔥 Ongoing Projects
<!--<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Product Recommendation</div><img src='images/r4.png' alt="sym" width="100%"></div></div>-->
<div class='paper-box'><div class='paper-box-image'><div><img src='images/r5.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

### **Estimating the Effect of Social Bots in Networks Using Counterfactual Graph with Adversarial Learning.**

**<font color="purple">Highlights</font>**
- Examines the impact difference between humans and bots in influencing opinions in social networks. 
- Designs an effective approach to conduct counterfactual inference in networks with graph neural networks and adversarial multitask learning.
- The impact differences between bots and humans are overestimated without considering the underlying homophily.

</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><img src='images/r6.jpg' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

### **Learning the Shifting Attention: A Novel Framework for Blockbuster Content Prediction.**

**<font color="purple">Highlights</font>**
- An ongoing project with an E-commerce company for conducting content selection and blockbuster prediction.
- Explore the internal and external factors fostering the generation of hotspots from the user attention perspective. 
- Design a theory-driven multivariate time series transfer learning approach to handle the shifting attention and user dynamics.

</div>
</div>



# 📝 Research Papers
### **Journal Papers**

**Ziyue Wu**, Xi Chen\*, Zhaoxing Gao. (2023). Bayesian Non-parametric Method for Decision Support: Forecasting Online Product Sales. ***Decision Support Systems***, 114019. [[Paper]](https://doi.org/10.1016/j.dss.2023.114019)

### **Conference Papers**

**Ziyue Wu**, Yiqun Zhang, Xi Chen\*. (2024). What if Social Bots Be My Friends? Estimating Causal Effect of Social Bots Using Counterfactual Graph Learning. 57th ***Hawaii International Conference on Systems Sciences (HICSS)***. [[Paper]](https://scholarspace.manoa.hawaii.edu/items/67acb13c-4e3e-43cd-bbc6-c43fbfdfb4a1)

# 💬 Working Papers

### **Uncovering the Long-tail: Market Membership-Aware Bipartite Graph Learning for Product Recommendation**

With Xi Chen, Ruijin Jin. ***Paper preparing for submission***.

**<font color="purple">Abstract:</font>** While superstar products account for the head of sales, long-tail products occupy major categories and can be highly lucrative. This study focuses on uncovering profitable long-tail products to enhance product recommendations. The primary challenge lies in the inadequacy of sparse interactions to accurately portray the qualities of long-tail products and capture their demand patterns. Furthermore, the superstar effect makes classical interaction-based recommendation algorithms concentrate on popular products and ignore long-tail ones. To address the issues, we propose a novel graph representation learning approach that leverages market structure information from the perspective of the product-customer relationships, which can (1) identify underlying sub-markets to enrich information regarding long-tail products, (2) model the heterogeneous demands within distinct sub-markets at a finer granularity, and (3) improve customer interaction prediction and product recommendations. Empirical and synthetic experiments demonstrate the superior prediction performance of our approach, and we also collaborate with a well-known e-commerce platform in China and conduct a field experiment to validate its real-world effectiveness in product recommendations. Our findings suggest the platform redesigning its categories by considering the composition of product submarkets and targeted customer segments to achieve higher profits.

---

### **Do Your Ties Affect Your Traits? A Graph Representation Learning Approach for Estimating Casual Peer Influence**

With Xi Chen. ***Paper preparing for submission***.

**<font color="purple">Abstract:</font>** Network data such as social relationships supports both theoretical research and practical applications in managerial studies. This study focuses on the challenge of unobservable homophilous sources in social networks, a key factor leading to confounding bias when examining the causal peer influence on individual behaviors. To tackle the issue, studies have attempted to leverage observable network ties to recover information about individual latent traits and control homophily. In this paper, we theoretically prove the insufficiency of structural information (i.e., network topology) and the essential of node contextual information for recovering individual traits in observational networks. Additionally, given the diverse factors influencing tie formation beyond homophily, we demonstrate that researchers should selectively leverage ties formed by endogenous factors. Building on these insights, we introduce CausalDG, a data-driven approach that enables us to (1) recover unobservable homophilous factors by incorporating both structural and contextual information in the absence of node features, and (2) adaptively leverage ties generated by endogenous factors under various network formation mechanisms to enhance the inference of individual latent traits and estimate the causal peer influence. We demonstrate the advantages of CausalDG in reducing the estimation bias of peer influence through extensive simulations. We also illustrate how CausalDG supports better decision-making using empirical networks. Moreover, our application and analysis in a large online social game underscore the capability of CausalDG to compensate for unavailable individual covariates when estimating peer influence.

---

### **Co-move in the Future? Stock Prediction with Dynamic Co-movement Graph**

With Xi Chen. ***Ongoing research paper***.

**<font color="purple">Abstract:</font>** Predicting future outcomes by uncovering transferable historical information in dynamic environments is a significant challenge in information systems research. This study focuses on the prediction of stock movements, a representative scenario characterized by high dynamics and a low signal-to-noise ratio. Existing research suggests that movement patterns of assets, such as stock returns, can be learned from related entities, such as the coordinated movements of other stock prices. However, historical dependency patterns may not persist in the future. Drawing on market signal theory and employing a design science approach, we propose utilizing stock co-movement information as a proxy task for stock price forecasting to distinguish endogenous and exogenous factors behind the observable dependencies and extract information capable of making generalizable forecasts. Our approach effectively handles complex long- and short-term co-movement patterns, outperforming state-of-the-art methods in forecasting stock movements and trading in Chinese stock markets. Our findings show that co-movement information can reveal common risks associated with stocks and possesses significant predictive ability. However, it is crucial to distinguish observable co-movements caused by exogenous shocks, as these patterns often undergo rapid changes.


# 📖 Educations
**2021.9 - 2026.6 (Expected)**, Ph.D. Candidate, Zhejiang University, School of Management. 

**2017.9 – 2021.6**, Undergraduate, Zhejiang University, Chu Kochen Honors College. 

# 🎉 Honors and Awards
2023.12 **<font color="purple">[Scholarship]</font>**, First-Class Scholarship from Zhejiang University & Transfer Group Co., Ltd (Top 2%).

2023.9 **<font color="purple">[Award]</font>**, Outstanding Ph.D. Candidate Award in School of Management, Zhejiang University (Top 5%).

2022.12 **<font color="purple">[Award]</font>**, Best student paper in Annual Conference of Big Data and Business Analytics and China Management Young Scholars Forum. 

2021.10 **<font color="purple">[Scholarship]</font>**, Freshman scholarship for Ph.D. Candidates (Top 5%). 

2019 **<font color="purple">[Scholarship]</font>**, Provincial Scholarship (Top 10%). 

2019 & 2020 **<font color="purple">[Award]</font>**, Research Innovation Pacesetter in Chu Kochen Honors College. 

2018 & 2019 **<font color="purple">[Award]</font>**, Outstanding Students in Chu Kochen Honors College (Top 10%). 

2018 & 2019 **<font color="purple">[Scholarship]</font>**, First-Class Scholarship for Excellent Students in Zhejiang University (Top 5%).

2018 **<font color="purple">[Scholarship]</font>**, National Scholarship (Top 2%).

# 🎖 Conferences and Talks
2024.1, What if Social Bots Be My Friends? Estimating Causal Effect of Social Bots Using Counterfactual Graph Learning. **Hawaii International Conference on Systems Sciences**, Honolulu, Hawaii, USA.

2023.12, Estimating the Impact of Social Bots in Opinion Diffusion in Social Networks. **Doctoral Forum in School of Management, Zhejiang University**, Hangzhou, Zhejiang, China.

2023.9, A Bayesian Non-parametric Method for Product Sales Forecasting. **Doctoral Forum in School of Management, Zhejiang University**, Hangzhou, Zhejiang, China.

2023.7, Leverage Market Membership Information for Long-tail Product Discovery. **China Management Annual Conference**, Urumqi, Xinjiang, China.

2022.12, A Network Representation Learning Approach to Identify Causality in Social Network Observational Data. **Annual Conference of Big Data and Business Analytics**, Changchun, Jilin, China.


# 💻 Activities
### **Exchange Experiences**

2023.10, Academic Visit and Exchange in **Tilburg University**, Netherlands.

2020.5 – 2020.9, Summer Intensity Research Training in the **University of California, Los Angeles (UCLA)**.

2018.7 – 2018.8, Lady Margaret Hall, **Oxford University**, UK (Best Presentation award for group academic project).


### **Services and Clubs**

2022.9 - 2023.9, Artificial Intelligence Association of Zhejiang University, Head of the Academic Department.

Started from 2021.11, Asia-Pacific Student Entrepreneurship Society, Member.

Started from 2020.5, Zhejiang University Morningside Culture China Scholars Program (12th), Member.


---

Last Modified on January 11, 2024
