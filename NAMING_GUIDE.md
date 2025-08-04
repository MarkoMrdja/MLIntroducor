# 📁 Document Naming Guide for Optimal Chunking

The system automatically detects document types based on filename keywords. Follow these naming conventions for best results:

## 🎯 **Optimal File Naming**

### **Exam Questions** 📝
**Keywords:** `ispitna`, `pitanja`, `ispit`, `questions`
**Chunking:** By topics (1., 2., 3., etc.)
```
✅ GOOD:
- ispitna_pitanja_2024.pdf
- pitanja_masinsko_ucenje.pdf  
- ispit_januar_2023.pdf

❌ AVOID:
- questions.pdf (too generic)
- test.pdf (ambiguous)
```

### **Exercise Sheets** 📚
**Keywords:** `vezbe`, `ponavljanje`, `zadaci`, `exercises`, `homework`
**Chunking:** By numbered problems
```
✅ GOOD:
- vezbe_linearna_algebra.pdf
- ponavljanje_verojatnoca.pdf
- zadaci_neural_networks.pdf

❌ AVOID:
- homework.pdf (too generic)
- problems.pdf (ambiguous)
```

### **Lecture Slides** 🎓
**Keywords:** `predavanje`, `slajd`, `prezentacija`, `lecture`, `slides`
**Chunking:** By concepts and bullet points
```
✅ GOOD:
- predavanje_01_uvod.pdf
- predavanje_parametarska_estimacija.pdf
- slajdovi_bayesova_teorija.pdf

❌ AVOID:
- slides.pdf (too generic)
- presentation.pdf (ambiguous)
```

### **Practicum/Books** 📖
**Keywords:** `praktikum`, `knjiga`, `skripta`, `teorija`, `textbook`
**Chunking:** Large semantic chunks
```
✅ GOOD:
- praktikum_masinsko_ucenje.pdf
- skripta_ml_teorija.pdf
- knjiga_pattern_recognition.pdf

❌ AVOID:
- book.pdf (too generic)
- material.pdf (ambiguous)
```

## 🔧 **Content-Based Fallback**

If filenames don't match patterns, the system analyzes content:
- **Numbered sections (1., 2., 3.)** → Exam questions
- **Bullet points (◼, ❑, •)** → Lecture slides  
- **Mathematical formulas** → Practicum
- **Short numbered items** → Exercises

## 📊 **Processing Results**

The system will show detection results:
```
🔍 Processing material: praktikum_masinsko_ucenje.pdf
🔍 Detected practicum in praktikum_masinsko_ucenje.pdf (by filename)
📄 Created 156 semantic chunks

🔍 Processing material: predavanje_neural_networks.pdf  
🔍 Detected lecture slides in predavanje_neural_networks.pdf (by filename)
📄 Created 23 concept-based chunks
```

## 💡 **Pro Tips**

1. **Use Serbian keywords** for better recognition
2. **Be specific** - include topic in filename
3. **Consistent naming** helps system learn your patterns
4. **Check processing logs** to verify correct detection

## 📁 **Recommended Directory Structure**

```
data/
├── material/
│   ├── praktikum_masinsko_ucenje_2024.pdf      # Main theory
│   ├── predavanje_01_uvod_u_ml.pdf             # Lecture 1
│   ├── predavanje_02_parametarska_est.pdf      # Lecture 2  
│   ├── predavanje_03_bayesova_teorija.pdf      # Lecture 3
│   ├── vezbe_linearna_algebra.pdf              # Math exercises
│   └── vezbe_verojatnoca_statistika.pdf        # Stats exercises
└── questions/
    ├── ispitna_pitanja_2024_januar.pdf         # Jan 2024 exam
    ├── ispitna_pitanja_2023_septembar.pdf      # Sep 2023 exam
    └── pitanja_kolokvijum_1.pdf                # Midterm questions
```

This naming ensures optimal chunking and makes your study system much more effective! 🎯
