// src/services/api.js
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'https://camp-mdxq.onrender.com/api';

const instance = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add a request interceptor to include X-User-Id if user is logged in
instance.interceptors.request.use(
  (config) => {
    const userString = localStorage.getItem('gomedcampUser');
    if (userString) {
      const user = JSON.parse(userString);
      if (user && user.id) {
        config.headers['X-User-Id'] = user.id;
      }
    }
    // If you use JWT tokens, you'd add Authorization header here too
    // const token = localStorage.getItem('gomedcampToken');
    // if (token) {
    //   config.headers['Authorization'] = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export default instance;
