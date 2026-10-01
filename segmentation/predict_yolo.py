import os
from ultralytics import YOLO


def main():
    # Carica i pesi salvati all'ultima epoca (32) del tuo addestramento
    # Ho corretto il percorso che per qualche motivo aveva un doppio "runs/segment"
    model = YOLO("runs/segment/runs/segment/bee_model_finetuned_yolo26s_dominiuniti/weights/best.pt")

    # Invece di una singola immagine, passiamo l'intera cartella di validazione!
    test_dir = "/mnt/disk1/borsattifr/datasets/bees_datasets/DatasetApi_Ceschi/processed/test_merged/images"
    
    # Assicuriamoci che la cartella esista prima di procedere
    if not os.path.exists(test_dir):
        print(f"⚠️ Cartella non trovata: {test_dir}.")
        return

    import cv2
    import glob
    import numpy as np
    import matplotlib.pyplot as plt

    print(f"Eseguendo il modello addestrato: {test_dir}")
    
    # Creiamo la cartella di output
    out_dir = "runs/segment/predict_resized_yolo26s_besteworst"
    os.makedirs(out_dir, exist_ok=True)
    
    # Processiamo le immagini una per una ridimensionandole
    image_paths = glob.glob(os.path.join(test_dir, "*.[jJ][pP][gG]")) + glob.glob(os.path.join(test_dir, "*.[pP][nN][gG]"))
    
    # Lista per raccogliere le statistiche di ogni immagine
    # Ogni elemento: (img_path, confidence_media, num_detection, immagine_con_predizioni)
    stats = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        MAX_DIM = 2000
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            
        # Predizione sull'immagine ridimensionata
        results = model.predict(source=img, save=False, show=False, conf=0.10, max_det=2000)        
        
        # Filtriamo le predizioni calcolando l'area MEDIANA delle api (come nel tuo pass1 di SAM)
        if len(results) > 0 and len(results[0].boxes) > 0:
            res = results[0]
            keep_indices = []
            
            # 1. Raccogliamo le aree di tutte le predizioni
            areas = []
            for box in res.boxes.xywh:
                w, h = box[2].item(), box[3].item()
                areas.append(w * h)
                
            # 2. Calcoliamo l'area mediana (ignorando così i picchi enormi o minuscoli)
            median_area = np.median(areas)
            
            # 3. La soglia minima sarà il x% dell'area media di un'ape in QUESTA specifica foto
            MIN_AREA = median_area * 0.6
            print(f"[{img_name}] Area mediana: {median_area:.0f} px -> Scarto tutto sotto i {MIN_AREA:.0f} px")
            
            for i, area in enumerate(areas):
                if area >= MIN_AREA:
                    keep_indices.append(i)
                    
            # Aggiorna i risultati mantenendo solo le api sufficientemente grandi
            res = res[keep_indices]
            
            if len(keep_indices) > 0:
                # Calcola la confidence media delle detection rimaste
                confidences = res.boxes.conf.cpu().numpy()
                mean_conf = float(np.mean(confidences))
                num_det = len(keep_indices)
                
                # labels=False nasconde la scritta "bee" e la percentuale
                # boxes=False nasconde il riquadro rettangolare, lasciando SOLO la maschera sagomata
                res_img = res.plot(labels=False, boxes=False)
                cv2.imwrite(os.path.join(out_dir, img_name), res_img)
                print(f"Salvato {img_name} in {out_dir} (conf media: {mean_conf:.3f}, detection: {num_det})")
                
                # Salva le statistiche
                stats.append({
                    "img_path": img_path,
                    "img_name": img_name,
                    "mean_conf": mean_conf,
                    "num_det": num_det,
                    "res_img": res_img,  # immagine con le predizioni plottate
                })
            else:
                print(f"[{img_name}] Nessuna detection rimasta dopo il filtraggio")
        else:
            print(f"[{img_name}] Nessuna detection trovata")
    
    print(f"\n✅ Inferenza completata! Processate {len(stats)} immagini con detection.")
    print(f"Controlla la cartella '{out_dir}' per vedere i risultati.")
    
    # ============================================================
    # Plot delle 10 migliori, 10 peggiori e 10 mediane
    # ============================================================
    if len(stats) < 10:
        print("⚠️ Meno di 10 immagini con detection, impossibile plottare le classifiche.")
        return
    
    # Ordina per confidence media (decrescente = migliore prima)
    stats_sorted = sorted(stats, key=lambda x: x["mean_conf"], reverse=True)
    
    top10 = stats_sorted[:10]
    bottom10 = stats_sorted[-10:]
    
    # Mediane: prendi le 10 immagini centrali
    n = len(stats_sorted)
    mid = n // 2
    start_median = max(0, mid - 5)
    end_median = min(n, start_median + 10)
    median10 = stats_sorted[start_median:end_median]
    
    def plot_grid(image_list, title, filename):
        """Plotta una griglia 2x5 di immagini con titolo e confidence."""
        fig, axes = plt.subplots(2, 5, figsize=(25, 12))
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        for idx, ax in enumerate(axes.flat):
            if idx < len(image_list):
                entry = image_list[idx]
                # Converti da BGR (OpenCV) a RGB (matplotlib)
                img_rgb = cv2.cvtColor(entry["res_img"], cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_title(
                    f"{entry['img_name']}\n"
                    f"Conf: {entry['mean_conf']:.3f} | Det: {entry['num_det']}",
                    fontsize=9
                )
            ax.axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Salvato: {save_path}")
    
    plot_grid(top10, "🏆 Top 10 — Confidence Media Più Alta", "top10_best.png")
    plot_grid(bottom10, "⚠️ Bottom 10 — Confidence Media Più Bassa", "top10_worst.png")
    plot_grid(median10, "📊 Median 10 — Confidence Media Mediana", "top10_median.png")
    
    print("\n📈 Statistiche generali:")
    all_confs = [s["mean_conf"] for s in stats]
    print(f"   Confidence media globale: {np.mean(all_confs):.3f}")
    print(f"   Confidence min: {np.min(all_confs):.3f}, max: {np.max(all_confs):.3f}")
    print(f"   Totale immagini processate: {len(stats)}")


if __name__ == "__main__":
    main()
