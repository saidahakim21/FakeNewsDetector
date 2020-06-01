# Flask Rest API for Fake News Detecor

This is an Api input for our fake news detector.

launch the program and then using Postman or any other REST client , launch the following query

        method : POST

        url : http://127.0.0.1:5000/detector

        body :

            {
                "headline": "YOUR HEADLINE",
                "body":  "YOUR BODY"
            }


No need to bother with the classifer, the folder already contains the trained model along side every binary object saved in *.pkl file