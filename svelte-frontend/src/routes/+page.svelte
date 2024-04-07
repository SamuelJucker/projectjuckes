<script>
    import { writable } from 'svelte/store';
  
    const prediction = writable(''); // Store for prediction result
  
    let formData = {
      open: 0.0,
      bid: 0.0,
      ask: 0.0,
      marketCap: 0.0,
      beta: 0.0,
      trailingPE: 0.0,
      volume: 0.0,
      modelResultPositive: 0.5, // Default value, adjust as needed
    };
  
    // Function to submit form data to the Flask API and update the prediction store with the response
    async function submitForm() {
      try {
        const response = await fetch('http://localhost:5000/api/predict?' + new URLSearchParams(formData));
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        prediction.set(data.prediction); // Update the prediction store with the result
      } catch (error) {
        console.error('Error fetching prediction:', error);
        prediction.set('Error fetching prediction. Please check console for details.');
      }
    }
  </script>
  
  <form on:submit|preventDefault={submitForm}>
    <input type="number" bind:value={formData.open} placeholder="Open Price" step="any">
    <input type="number" bind:value={formData.bid} placeholder="Bid" step="any">
    <input type="number" bind:value={formData.ask} placeholder="Ask" step="any">
    <input type="number" bind:value={formData.marketCap} placeholder="Market Cap" step="any">
    <input type="number" bind:value={formData.beta} placeholder="Beta" step="any">
    <input type="number" bind:value={formData.trailingPE} placeholder="Trailing PE" step="any">
    <input type="number" bind:value={formData.volume} placeholder="Volume" step="any">
    <input type="number" bind:value={formData.modelResultPositive} placeholder="Sentiment Score" step="any">
    <button type="submit">Predict</button>
  </form>
  
  {#if $prediction}
    <p>Prediction: {$prediction}</p>
  {/if}
  