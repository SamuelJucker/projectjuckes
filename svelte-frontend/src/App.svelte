<script>
	import { writable } from 'svelte/store';
	
	const prediction = writable('');
	
	let formData = {
	  open: 0.0,
	  bid: 0.0,
	  ask: 0.0,
	  marketCap: 0.0,
	  beta: 0.0,
	  trailingPE: 0.0,
	  volume: 0.0,
	  modelResultPositive: 0.5,
	};
	
	async function submitForm() {
	  try {
		const response = await fetch('http://localhost:5000/api/predict?' + new URLSearchParams(formData));
		if (!response.ok) {
		  throw new Error('Network response was not ok');
		}
		const data = await response.json();
		prediction.set(data.prediction);
	  } catch (error) {
		console.error('Error fetching prediction:', error);
		prediction.set('Error fetching prediction. Please check console for details.');
	  }
	}
  </script>
  
  <h1>Stock Market Prediction</h1>
  <p>Enter the details below to predict the stock market behavior.</p>
  
  <form on:submit|preventDefault={submitForm}>
	<div>
	  <label for="open">Open Price:</label>
	  <input id="open" type="number" bind:value={formData.open} placeholder="Open Price" step="any">
	</div>
  
	<div>
	  <label for="bid">Bid:</label>
	  <input id="bid" type="number" bind:value={formData.bid} placeholder="Bid" step="any">
	</div>
  
	<div>
	  <label for="ask">Ask:</label>
	  <input id="ask" type="number" bind:value={formData.ask} placeholder="Ask" step="any">
	</div>
  
	<div>
	  <label for="marketCap">Market Cap:</label>
	  <input id="marketCap" type="number" bind:value={formData.marketCap} placeholder="Market Cap" step="any">
	</div>
  
	<div>
	  <label for="beta">Beta:</label>
	  <input id="beta" type="number" bind:value={formData.beta} placeholder="Beta" step="any">
	</div>
  
	<div>
	  <label for="trailingPE">Trailing PE:</label>
	  <input id="trailingPE" type="number" bind:value={formData.trailingPE} placeholder="Trailing PE" step="any">
	</div>
  
	<div>
	  <label for="volume">Volume:</label>
	  <input id="volume" type="number" bind:value={formData.volume} placeholder="Volume" step="any">
	</div>
  
	<div>
	  <label for="sentiment">Sentiment Score:</label>
	  <input id="sentiment" type="number" bind:value={formData.modelResultPositive} placeholder="Sentiment Score" step="any">
	</div>
  
	<button type="submit">Predict</button>
  </form>
  
  
  {#if $prediction}
	<p>Prediction: {$prediction}</p>
  {/if}
  
  <style>
	form div {
	  margin-bottom: 1rem;
	}
  
	input, button {
	  margin-top: 0.5rem;
	  padding: 0.5rem;
	  width: 100%;
	}
  
	label {
	  display: block;
	}
	h1 {
	  color: #ff3e00;
	}
	form {
	  margin: 20px 0;
	}
	input, button {
	  margin: 5px;
	  padding: 10px;
	}
  </style>

