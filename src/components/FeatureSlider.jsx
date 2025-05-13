// src/components/FeatureSlider.jsx
import React from 'react';
import Slider from 'react-slick';
import FeatureCard from './FeatureCard';
import { features } from '../featuresData'; // Import your features
import { ChevronLeft, ChevronRight } from 'lucide-react';

const NextArrow = (props) => {
  const { className, style, onClick } = props;
  return (
    <div className={className} style={{ ...style, display: 'block' }} onClick={onClick}>
      <ChevronRight size={30} />
    </div>
  );
};

const PrevArrow = (props) => {
  const { className, style, onClick } = props;
  return (
    <div className={className} style={{ ...style, display: 'block' }} onClick={onClick}>
      <ChevronLeft size={30} />
    </div>
  );
};

const FeatureSlider = () => {
  const settings = {
    dots: true,
    infinite: true,
    speed: 700, // Transition speed
    slidesToShow: 3, // Show 3 cards at a time
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 3000, // Time between slides
    pauseOnHover: true,
    nextArrow: <NextArrow />,
    prevArrow: <PrevArrow />,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 1,
          infinite: true,
          dots: true
        }
      },
      {
        breakpoint: 600,
        settings: {
          slidesToShow: 1,
          slidesToScroll: 1
        }
      }
    ]
  };

  return (
    <div className="feature-slider-container">
      <Slider {...settings}>
        {features.map(feature => (
          <FeatureCard key={feature.id} feature={feature} />
        ))}
      </Slider>
    </div>
  );
};

export default FeatureSlider;
