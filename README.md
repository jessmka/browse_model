# Abandoned Browse product email feature
This feature selects a product from a users pageviews to send a reminder email to them. It runs hourly taking in the last 24 hours of pageviews and decides if a user will receive the email. If they meet the threshold, or are randomly selected, the model will choose the highest affinity product that has inventory from their pageviews and sends an object to a webhook that will dynamically populate the email template with the chose product
Inputs:
  * Pageviews
  * Product attributes
  * Inventory

## Models
1. Logistic model: This model decides if a user will receive the email. If they have a low probability of purchase they are held back until the model is run again.
2. Product affinity: This uses the Spotlight package, which makes sequential recommendations and returns an affinity score for each product. This is filtered on products they have actually viewed, then inventory is checked and the top scored product with inventory is sent
