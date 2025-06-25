const express = require('express');
const fetch = require('node-fetch');
const app = express();

const API_TOKEN = '7Z1o3XulyYXsxnVtIU77zmooLSMQzGW6ksONP329BUdK7yl8lwSSrwDrkeSW'; // Replace with your actual token

app.get('/fixture/:fid', async (req, res) => {
  const fid = req.params.fid;
  const url = `https://cricket.sportmonks.com/api/v2.0/fixtures/${fid}?include=localteam,visitorteam,batting,balls&api_token=${API_TOKEN}`;

  try {
    const response = await fetch(url);
    if (!response.ok) return res.status(response.status).json({ error: 'Upstream API error' });

    const data = await response.json();
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(3000, () => console.log('Proxy server running at http://localhost:3000'));
