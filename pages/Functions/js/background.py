import streamlit as st

particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  html, body {
    margin: 0;
    padding: 0;
    overflow: hidden;
    width: 100%;
    height: 100%;
  }
  
  #particles-js {
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #000;
    z-index: -1;
  }
  
  /* 确保canvas可以接收鼠标事件 */
  canvas {
    pointer-events: auto !important;
  }
</style>
</head>
<body>
  <div id="particles-js"></div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    window.onload = function() {
      initParticles();
    };
    
    function initParticles() {
      particlesJS("particles-js", {
        "particles": {
          "number": {
            "value": 80,
            "density": {
              "enable": true,
              "value_area": 800
            }
          },
          "color": {
            "value": "#ffffff"
          },
          "shape": {
            "type": "circle",
            "stroke": {
              "width": 0,
              "color": "#000000"
            },
            "polygon": {
              "nb_sides": 5
            }
          },
          "opacity": {
            "value": 0.5,
            "random": false,
            "anim": {
              "enable": false,
              "speed": 1,
              "opacity_min": 0.1,
              "sync": false
            }
          },
          "size": {
            "value": 3,
            "random": true,
            "anim": {
              "enable": false,
              "speed": 40,
              "size_min": 0.1,
              "sync": false
            }
          },
          "line_linked": {
            "enable": true,
            "distance": 150,
            "color": "#ffffff",
            "opacity": 0.4,
            "width": 1
          },
          "move": {
            "enable": true,
            "speed": 2,
            "direction": "none",
            "random": false,
            "straight": false,
            "out_mode": "out",
            "bounce": false,
            "attract": {
              "enable": false,
              "rotateX": 600,
              "rotateY": 1200
            }
          }
        },
        "interactivity": {
          "detect_on": "canvas",
          "events": {
            "onhover": {
              "enable": true,
              "mode": "grab"
            },
            "onclick": {
              "enable": true,
              "mode": "push"
            },
            "resize": true
          },
          "modes": {
            "grab": {
              "distance": 140,
              "line_linked": {
                "opacity": 1
              }
            },
            "bubble": {
              "distance": 400,
              "size": 40,
              "duration": 2,
              "opacity": 8,
              "speed": 3
            },
            "repulse": {
              "distance": 200,
              "duration": 0.4
            },
            "push": {
              "particles_nb": 4
            },
            "remove": {
              "particles_nb": 2
            }
          }
        },
        "retina_detect": true
      });

      function resizeCanvas() {
        const canvas = document.querySelector('#particles-js canvas');
        if (canvas) {
          canvas.width = window.innerWidth;
          canvas.height = window.innerHeight;
          
          canvas.style.pointerEvents = 'auto';
        }
      }
      
      resizeCanvas();
      
      window.addEventListener('resize', resizeCanvas);
      
      try {
        window.parent.postMessage('particles-loaded', '*');
      } catch (e) {
        console.error('无法发送消息到父窗口:', e);
      }
      
      console.log('粒子效果已初始化');
    }
  </script>
</body>
</html>
"""


def particles():
    st.markdown("""
    <style>
        .stApp {
            background-color: transparent !important;
        }

        .main .block-container {
            background-color: rgba(0, 0, 0, 0.3) !important;
            border-radius: 10px;
            padding: 20px;
            margin-top: 10px;
        }

        .css-1d391kg, .css-1lcbmhc, .st-emotion-cache-1wbqy5l {
            background-color: rgba(0, 0, 0, 0.3) !important;
        }

        /* 确保粒子背景在所有内容后面 */
        iframe {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            z-index: -1 !important;
            border: none !important;
        }

        /* 添加一个透明层来接收鼠标事件，但允许它们传递到iframe */
        .content-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: 0;
            pointer-events: none;
        }

        /* 确保聊天内容在粒子上方 */
        .stChatMessage, .stChatInput {
            background-color: rgba(0, 0, 0, 0.5) !important;
            border-radius: 8px;
            margin: 5px 0;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.components.v1.html(particles_js, height=2000, width=2000)
    st.markdown('<div class="content-overlay"></div>', unsafe_allow_html=True)
