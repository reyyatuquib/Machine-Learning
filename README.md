# My project

<h1> A machine learning algorithm, which is able to predict if a customer will buy again</h1>

Given a data from an Audiobook app. Logically, it relates only to the audio versions of books.
Each customer in the database has made a purchase at least once, that's why he/she is in the database. We want to create a machine learning algorithm based on our available data that can predict if a customer will buy again from the Audiobook company.

The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertizing to him/her. If we can focus our efforts ONLY on customers that are likely to convert again, we can make great savings. Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers creates value and growth opportunities.

The targets are a Boolean variable (so 0, or 1). We are taking a period of 2 years in our inputs, and the next 6 months as targets. So, in fact, we are predicting if: based on the last 2 years of activity and engagement, a customer will convert in the next 6 months. 6 months sounds like a reasonable time. If they don't convert after 6 months, chances are they've gone to a competitor or didn't like the Audiobook way of digesting information.

All three set are balanced ( Train, Validation and Test) as we obtained mostly 50% of the result. 

You can see the code on my project "Audio Books_preprocessed data.py"
<img src="https://user-images.githubusercontent.com/115962820/197748237-4c05b4c5-6923-47ae-804c-f50596fee844.png" style="float:left;width:420px;height:auto;">



<img src="https://github.com/reyyatuquib/myproject/blob/main/assets/WIN_20211014_15_00_40_Pro.jpg" style="float:left;width:420px;height:auto;">
