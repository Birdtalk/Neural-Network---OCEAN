<header>
  <h1  align="left">Machine Learning Research, OCEAN (Myers-Briggs) user categorization with Neural Network Models</h1>
</p>
  <p>
    This research was performed under the supervision of Dr. Razvan Andonie at Central Washington University. The goal being create a neural network model which had some application to categorize users of social media inside of a standard OCEAN model especially in mind of Myers-Briggs categories. 
  </p>
</header>


<!-- table of contents-->
<nav>
      <hr>
      <p align="left">
	    <a href="#inspiration">Inspiration</a> •
         <a href="#goals">Goals</a> •
            <a href="#key-features">Key Features</a> •
            <a href="#how-to-use">How it Works</a> 
      </p>

</nav>
<section id="demonstration">
    <h1>Demonstration</h1>
<p>Please Click the video link below to watch a brief demonstration of the project.</p>
	<br/>
<a href="http://www.youtube.com/watch?feature=player_embedded&v=vzJW4M5YSiI" target="_blank">
 <img src="http://img.youtube.com/vi/vzJW4M5YSiI/mqdefault.jpg" alt="Watch the video" width="480" height="360" border="10" />
</a>

</section>
<section id="inspiration">

  <h1>Research Inspiration</h1>
  <p>
    This research was inspired by natural language processing in machine learning, which allows a computer model to derive meaning from words. Fascinating! Although there has already been absolute mountains of reserach done in this area my research would focus on using statistically consequential words to arrive at scores for the standard OCEAN model ( Openness, Conscientiousness, Extroversion, Agreeableness, and Neuroticism ). Much of my work is inspired by 3 sources: 
    <ol>
      <li>Andrew Dunn, "Sentiment Analysis of Tweets", Central Washington University (Ellensburg, Washington, United States) </li>
      <li>Hima Vijay & Neenu Sebastian, "Personality Prediction using Machine Learning", SCMS School of Engineering & Technology -  (Vidya Nagar, Karukutty, Ernakulam)</li>
      <li>Tal Yarkoni, "Personality in 100,000 Words: A large-scale analysis of personality and word use among bloggers"</li>
    </ol>
    A great deal of the ligitimacy that this method holds is due to the statistical work performed by Tal Yarkoni performed in 2010.
      <div align="center">
    <img src="talyankoni.png" width="400px"</img> 
	
</div>

The research of Andrew Dunn was referenced for specific methods for collecting, cleaning, and using data. The research of Hima Vijay and Neenu Sebastian was instrumental for constructing the neural network model. The research of Tal Yarkoni was used for the statistical data surrounding the weighting of words in reference to OCEAN model score assignment.
  </p>
  <br/>
</section>

<section id="goals">
  <h1>Goals</h1>
  <p>
    
  <ol>
    <li>
     1. Create a neural Network model that is capable of intaking a small number of words (150 words) and create an OCEAN profile based on those words. Fine tuning the "Bag-of-Words" is vital in this natural language method. The accuracy of all results is based on the weight given to the words contained in this dictionary.
	      <div align="center">
    <img src="someresults.png" width="400px"</img> 
</div>
    </li>
    <li>
      2. Focus on accuracy and achieve interpolation approximating OCEAN score distribution measured naturally. My results were tuned to mimic the distribution of broader OCEAN scores in the population.
	      <div align="center">
    <img src="compbars.png" width="400px"</img> 
</div>
    </li>
    <li>
      3. Write a appropriate report discussing my research, methods, findings, and reflections.
    </li>
    <li>
      4. Create a presentation to represent my the journey of research in an open, educational, and entertaining way.
    </li>
    <li>
      5. A demonstration video to tour some code and show the efficacy of the model.
    </li>
  </ol>

    
  </p>
  <br/>
</section>

<section id="key-features">
  <!-- Demonstration GIF -->
  <article>
    <h1>Key Features</h1>


      
- [x] A trained Neural Network model implemented in software
- [x] Utilize real text using the Twitter API to categorize real social media users.
- [x] A useful tool for analyzing behavior of Twitter users and apply context to controversial tweets

- [x] Accurately assess natural language 


  </article>
  <br/>
</section>

<section id="how-to-use">
  <article>
    <h1>How it Works</h1>
    <p>Use the categorization power of my premade machine learning model to input text and get an OCEAN score for the user who created that text. The score will be up to 70% and the more text that is input the better the accuracy becomes!</p>

 


  </article>
  <br/>
</section>

