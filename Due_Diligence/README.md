# Due Diligence Analysis On Fast-Food Chains

In this project, I wanted to run an outside-in diligence analysis with transaction-level data across nine different restaurant chains. The goal is the be able to identify which companies look financially healthiest for a potential investment.
I am personally interested in the applications of data science and data anlysis in the context of alternative investments like Private Equity. I believe that leveraging alternative data like transaction-level data shown here, when combined with overall company financials such as revenue, margins, EBITDA and free cash flows can help drive smarter and better investment decisions.

The data looks at transaction information from 9 different restaurant chains: Chipotle, PaneraBread, Dunkin Donuts, Wingstop, Subway, Taco Bell, Jimmy John’s, Sweetgreen, Shake Shack and Burger King. The dataset spans transactions that occurred between 2019 and 2022.
It can be found in this [folder](https://github.com/ndhers/My-Portfolio/blob/main/Due_Diligence/data/). 
Each of the four tables serves a different purpose:
<ul>
<li>Qsr_case_study_transactions_20230504 looks at order information. For example, it includes information about the order price, discount, item bought and quantity.</li> 
<li>Qsr_case_study_merchants_20230504 looks at restaurant chain information. This provides more information into what each merchant sells.</li>
<li>Qsr_case_study_users_20230504 looks at customer information. This provides information on users who have placed an order and includes data such as the state they live in, their gender, their income bracket, etc.</li>
<li>Qsr_case_study_descriptions_20230504 provides descriptions of the items being bought and sold.</li>
</ul>

The code for the analysis is included in this [notebook](https://github.com/ndhers/My-Portfolio/blob/main/Due_Diligence/main.ipynb). An excel sheet with the key output and figures is found [here](https://github.com/ndhers/My-Portfolio/blob/main/Due_Diligence/output.xlsx).
Finally, a written report for communicating the results in a clear and less technical manner is found in this [document](https://github.com/ndhers/My-Portfolio/blob/main/Due_Diligence/Summary_and_Takeaways.docx). 

I analyzed metrics such as quarterly order volume market share and quarterly revenue market share broken down by state and by household income group. Metrics such as customer overlap and repurchase rate for
different cohorts are also computed and leveraged for a complete analysis. Trends over the past couple years are analyzed to understand past behavior and make guesses on the future. Below is an example taken from the analysis where we see the percentage of customer overlap in 2022. 

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/blob/cust_overlap.png)

And here is the cohort analysis output for Chipotle demonstrating repurchase rate for different 'join-in' cohorts:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/blob/cohort.png)

Even though Dunkin Donuts' brand is very strong and the chain has shown increasing revenue market share over the past quarters, my final recommendation is Chipotle. Given the results shown in the output excel [file](https://github.com/ndhers/My-Portfolio/blob/main/Due_Diligence/output.xlsx) and
[report](https://github.com/ndhers/My-Portfolio/blob/main/Due_Diligence/Summary_and_Takeaways.docx), Chipotle dominates its counterparts in all of revenue market share, order volume market share and total number of customers. Chipotle is also repeatedly among the top restaurant chains in revenue market shares within those states that brought most overall revenue.
It is the leading restaurant chain by revenue market share in all of the household income brackets.
I also demonstrated the quality of user experience at Chipotle, which has maintained the second highest retention rate among its competitors, only beaten by Dunkin Donuts. This shows great ability at finding ways to keeping new customers excited and interested in their products. 
Finally, through the Customer Overlap analysis, it was shown that Chipotle possesses the strongest brand values and customer loyalty. It is able to maintain the highest number of exclusive customers, while keeping percentages of its customers that also consume at other chains low. 




