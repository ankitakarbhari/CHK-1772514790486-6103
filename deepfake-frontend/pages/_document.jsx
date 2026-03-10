import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en" className="dark">
      <Head>
        {/* Favicon */}
        <link rel="icon" href="/favicon.ico" />
        
        {/* Fonts */}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
        <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet" />
        
        {/* Meta tags */}
        <meta name="description" content="AI-Powered Deepfake Detection System" />
        <meta name="theme-color" content="#030305" />
      </Head>
      <body className="bg-dark-500 text-gray-100 font-sans antialiased">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}