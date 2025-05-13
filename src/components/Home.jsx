// src/components/Home.jsx
import React from 'react';
import FeatureSlider from './FeatureSlider';

const Home = () => {
  return (
    <div>
      <h1>Welcome to GoMedCamp</h1>
      <p style={{textAlign: 'center', marginBottom: '2rem'}}>Innovative solutions for accessible healthcare.</p>
      <FeatureSlider />
      {/* You can add more content to the home page here */}
    </div>
  );
};

export default Home;
