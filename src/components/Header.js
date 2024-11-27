import React from "react";
import styled from "styled-components";
import LogoImage from "../assets/logo.png"; 

const Header = ({ onBack }) => {
  return (
    <HeaderContainer>
      <BackButton onClick={onBack}>
        <Arrow>←</Arrow>
      </BackButton>
      <TitleContainer>
        <Logo src={LogoImage} alt="SignBridge Logo" />
        <Title>SignBridge</Title>
      </TitleContainer>
    </HeaderContainer>
  );
};

export default Header;

const HeaderContainer = styled.div`
  background-color: #ffffff;
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e0e0e0;
`;

const BackButton = styled.div`
  background-color: #4a53ff; /* Purple color from style guide */
  color: #ffffff;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
`;

const Arrow = styled.span`
  font-size: 1.5rem;
  font-weight: bold;
`;

const TitleContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  flex-grow: 1;
`;

const Logo = styled.img`
  width: 40px;
  height: auto;
  margin-right: 0.5rem;
`;

const Title = styled.h1`
  font-family: "Quicksand", sans-serif;
  font-size: 1.5rem;
  color: #333333;
  margin: 0;
  text-align: center;
`;
