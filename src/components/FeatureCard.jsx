// src/components/FeatureCard.jsx
import React from 'react';

const FeatureCard = ({ feature }) => {
  return (
    <div className="feature-card"> {/* This is the slide item for react-slick */}
      <img src={feature.image} alt={feature.title} className="feature-card-image" />
      <div className="feature-card-content">
        <h3>{feature.title}</h3>
        <p>{feature.description}</p>
      </div>
    </div>
  );
};

export default FeatureCard;
